use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tokio::sync::oneshot;
use tracing::{error, info, warn};

use baseten_performance_client_core::http;

use crate::config::TokenizerConfig as ProxyTokenizerConfig;
use tokenizers::Tokenizer as HuggingFaceTokenizer;

/// Tokenizer worker request types
#[derive(Debug)]
pub enum TokenizerRequest {
    DecodeBatch(Vec<Vec<u32>>, oneshot::Sender<Result<Vec<String>, String>>),

    Shutdown,
}

/// Handle for interacting with a tokenizer worker
#[derive(Debug, Clone)]
pub struct TokenizerHandle {
    name: String,
    tx: async_channel::Sender<TokenizerRequest>,
}

impl TokenizerHandle {
    /// Batch decode multiple token sequences
    pub async fn decode_batch(&self, token_sequences: &[Vec<u32>]) -> Result<Vec<String>, String> {
        let (response_tx, response_rx) = oneshot::channel();

        if let Err(e) = self
            .tx
            .send(TokenizerRequest::DecodeBatch(
                token_sequences.to_vec(),
                response_tx,
            ))
            .await
        {
            return Err(format!(
                "Failed to send decode batch request to tokenizer '{}': {}",
                self.name, e
            ));
        }

        match response_rx.await {
            Ok(result) => result,
            Err(e) => Err(format!(
                "Failed to receive decode batch response from tokenizer '{}': {}",
                self.name, e
            )),
        }
    }
}

/// Tokenizer worker thread
struct TokenizerWorker {
    name: String,
    config: ProxyTokenizerConfig,
    tokenizer: Option<HuggingFaceTokenizer>,
    shutdown: Arc<AtomicBool>,
}

impl TokenizerWorker {
    fn new(name: String, config: ProxyTokenizerConfig, shutdown: Arc<AtomicBool>) -> Self {
        Self {
            name,
            config,
            tokenizer: None,
            shutdown,
        }
    }

    /// Run the worker thread
    fn run(mut self, rx: async_channel::Receiver<TokenizerRequest>) {
        info!("Starting tokenizer worker: {}", self.name);

        // Initialize tokenizer
        if let Err(e) = self.initialize_tokenizer() {
            error!("Failed to initialize tokenizer '{}': {}", self.name, e);
            return;
        }

        info!("Tokenizer worker '{}' initialized successfully", self.name);

        // Process requests using async_channel with blocking recv
        while !self.shutdown.load(Ordering::SeqCst) {
            match rx.recv_blocking() {
                Ok(request) => {
                    if let Err(e) = self.handle_request(request) {
                        error!("Error handling request in tokenizer '{}': {}", self.name, e);
                    }
                }
                Err(_) => {
                    info!("Tokenizer worker '{}' channel disconnected", self.name);
                    break;
                }
            }
        }

        info!("Tokenizer worker '{}' shutting down", self.name);
    }

    /// Initialize the tokenizer
    fn initialize_tokenizer(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let tokenizer = if self.config.tokenizer_path.exists() {
            // Load from file if it exists
            HuggingFaceTokenizer::from_file(&self.config.tokenizer_path).map_err(|e| {
                error!(
                    "Failed to load tokenizer from file {}: {}",
                    self.config.tokenizer_path.display(),
                    e
                );
                e
            })?
        } else {
            // Try to load from pretrained model
            warn!(
                "Tokenizer file not found, attempting to load pretrained model: {}",
                self.config.model_id
            );
            HuggingFaceTokenizer::from_file(&self.config.model_id).map_err(|e| {
                error!(
                    "Failed to load pretrained tokenizer {}: {}",
                    self.config.model_id, e
                );
                e
            })?
        };

        info!(
            "Successfully loaded tokenizer: {} (vocab size: {})",
            self.name,
            tokenizer.get_vocab_size(true)
        );
        self.tokenizer = Some(tokenizer);
        Ok(())
    }

    /// Handle a request from the channel
    fn handle_request(
        &mut self,
        request: TokenizerRequest,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match request {
            TokenizerRequest::DecodeBatch(token_sequences, response_tx) => {
                let result = if let Some(ref tokenizer) = self.tokenizer {
                    // Convert Vec<Vec<u32>> to Vec<&[u32]> for the tokenizer
                    let token_refs: Vec<&[u32]> =
                        token_sequences.iter().map(|seq| seq.as_slice()).collect();
                    tokenizer
                        .decode_batch(&token_refs, true)
                        .map_err(|e| format!("Decode batch error: {}", e))
                } else {
                    Err("Tokenizer not initialized".to_string())
                };

                if let Err(e) = response_tx.send(result) {
                    error!("Failed to send decode batch response: {:?}", e);
                }
            }

            TokenizerRequest::Shutdown => {
                info!("Tokenizer worker '{}' received shutdown signal", self.name);
                self.shutdown.store(true, Ordering::SeqCst);
            }
        }
        Ok(())
    }
}

/// Tokenizer manager configuration
#[derive(Debug, Clone)]
pub struct TokenizerManagerConfig {
    pub tokenizers: HashMap<String, ProxyTokenizerConfig>,
    pub default_tokenizer: Option<String>,
    pub channel_buffer: usize,
    pub shutdown_timeout_ms: u64,
}

impl Default for TokenizerManagerConfig {
    fn default() -> Self {
        Self {
            tokenizers: HashMap::new(),
            default_tokenizer: None,
            channel_buffer: 1000,
            shutdown_timeout_ms: 100,
        }
    }
}

impl TokenizerManagerConfig {
    /// Create from proxy config
    pub fn from_proxy_config(proxy_config: &crate::config::ProxyConfig) -> Self {
        Self {
            tokenizers: proxy_config.tokenizers.clone(),
            default_tokenizer: proxy_config.tokenizers.keys().next().cloned(),
            channel_buffer: 1000,
            shutdown_timeout_ms: 100,
        }
    }
}

/// Tokenizer manager with channel-based architecture
pub struct TokenizerManager {
    config: TokenizerManagerConfig,
    handles: HashMap<String, TokenizerHandle>,
    shutdown: Arc<AtomicBool>,
    is_initialized: bool,
}

impl TokenizerManager {
    /// Create a new tokenizer manager
    pub fn new(config: TokenizerManagerConfig) -> Self {
        Self {
            config,
            handles: HashMap::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
            is_initialized: false,
        }
    }

    /// Initialize all tokenizer workers
    pub async fn initialize(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.is_initialized {
            info!("Tokenizer manager already initialized");
            return Ok(());
        }

        info!(
            "Initializing tokenizer manager with {} tokenizers",
            self.config.tokenizers.len()
        );

        // Spawn one worker thread per tokenizer
        for (name, config) in &self.config.tokenizers {
            let (tx, rx) = async_channel::bounded(self.config.channel_buffer);
            let shutdown = self.shutdown.clone();
            let name_clone = name.clone();
            let config = config.clone();

            thread::spawn(move || {
                let worker = TokenizerWorker::new(name_clone, config, shutdown);
                worker.run(rx);
            });

            let handle = TokenizerHandle {
                name: name.clone(),
                tx,
            };

            self.handles.insert(name.clone(), handle);
            info!("Spawned tokenizer worker: {}", name);
        }

        self.is_initialized = true;
        info!("Tokenizer manager initialization completed");
        Ok(())
    }

    /// Get a tokenizer handle by name
    pub async fn get_tokenizer(&self, name: &str) -> Result<TokenizerHandle, String> {
        if !self.is_initialized {
            return Err("Tokenizer manager not initialized".to_string());
        }

        self.handles.get(name)
            .cloned()
            .ok_or_else(|| {
                let available = self.handles.keys().cloned().collect::<Vec<_>>();
                if available.is_empty() {
                    format!("Tokenizer '{}' not found. No tokenizers are configured. Use --tokenizer <model_id> <path> to configure tokenizers.", name)
                } else {
                    format!("Tokenizer '{}' not found. Available tokenizers: {}", name, available.join(", "))
                }
            })
    }

    /// List all available tokenizer names
    pub async fn list_tokenizers(&self) -> Vec<String> {
        self.handles.keys().cloned().collect()
    }

    /// Check if a tokenizer exists
    pub async fn has_tokenizer(&self, name: &str) -> bool {
        self.handles.contains_key(name)
    }

    /// Check if a model exists and return available models if not found
    pub async fn has_model(&self, model_name: &str) -> Result<bool, String> {
        if !self.is_initialized {
            return Err("Tokenizer manager not initialized".to_string());
        }

        if self.handles.contains_key(model_name) {
            Ok(true)
        } else {
            let available_models = self.handles.keys().cloned().collect::<Vec<_>>();
            if available_models.is_empty() {
                Err("No tokenizers are configured. Use --tokenizer <model_id> <path> to configure tokenizers.".to_string())
            } else {
                Err(format!(
                    "Model '{}' not found. Available models: {}",
                    model_name,
                    available_models.join(", ")
                ))
            }
        }
    }

    /// Process tokenized embeddings request
    pub async fn process_tokenized_request(
        &self,
        request: &crate::schema::TokenizedEmbeddingsRequest,
        tokenizer_name: Option<&str>,
    ) -> Result<http::CoreOpenAIEmbeddingsRequest, String> {
        // Determine which tokenizer to use
        let tokenizer_name = match tokenizer_name {
            Some(name) => name.to_string(),
            None => {
                // Try to get the first available tokenizer
                if let Some((name, _)) = self.handles.iter().next() {
                    info!("Using first available tokenizer: {}", name);
                    name.clone()
                } else {
                    return Err("No tokenizer specified and no default available. Use --tokenizer <model_id> <path> to configure tokenizers.".to_string());
                }
            }
        };

        let handle = self.get_tokenizer(&tokenizer_name).await?;

        // Decode tokenized inputs to strings
        let decoded_inputs = handle
            .decode_batch(&request.input)
            .await
            .map_err(|e| format!("Failed to decode inputs: {}", e))?;

        // Create core embeddings request
        Ok(http::CoreOpenAIEmbeddingsRequest {
            input: decoded_inputs,
            model: request.model.clone(),
            encoding_format: request.encoding_format.clone(),
            dimensions: request.dimensions,
            user: request.user.clone(),
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &TokenizerManagerConfig {
        &self.config
    }
}

/// Implement graceful shutdown
impl Drop for TokenizerManager {
    fn drop(&mut self) {
        info!("Shutting down tokenizer manager...");
        self.shutdown.store(true, Ordering::SeqCst);

        // Send shutdown signals to all workers
        for handle in self.handles.values() {
            std::mem::drop(handle.tx.send(TokenizerRequest::Shutdown));
        }

        // Give workers time to shut down gracefully
        std::thread::sleep(Duration::from_millis(self.config.shutdown_timeout_ms));

        info!("Tokenizer manager shutdown completed");
    }
}

/// Create a tokenizer manager from proxy configuration
pub fn create_tokenizer_manager_from_proxy_config(
    proxy_config: &crate::config::ProxyConfig,
) -> TokenizerManager {
    let config = TokenizerManagerConfig::from_proxy_config(proxy_config);
    TokenizerManager::new(config)
}

use serde::{Deserialize, Serialize};

/// Tokenized embeddings request structure
/// This accepts tokenized inputs (list[uint32]) instead of strings
/// The tokenized inputs will be detokenized back to strings before being sent upstream
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenizedEmbeddingsRequest {
    /// Tokenized inputs - each input is a list of token IDs (uint32)
    /// This will be detokenized back to strings before processing
    pub input: Vec<Vec<u32>>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

use axum::{
    extract::{Request, State},
    http::{Method, StatusCode},
    response::{IntoResponse, Json},
    routing::{any, get},
    Router,
};
use baseten_performance_client_core::HttpMethod;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::signal;
use tower::ServiceBuilder;
use tower_http::cors::{Any, CorsLayer};
use tracing::{debug, error, info, warn};

use crate::config::ProxyConfig;
use crate::constants;
use crate::handlers::UnifiedHandler;
use crate::tokenizer_manager::create_tokenizer_manager_from_proxy_config;

pub async fn create_server(config: Arc<ProxyConfig>) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tokenizer manager
    let tokenizer_manager = if !config.tokenizers.is_empty() {
        let mut manager = create_tokenizer_manager_from_proxy_config(&config);
        info!(
            "Loading tokenizer manager with {} tokenizers...",
            config.tokenizers.len()
        );
        if let Err(e) = manager.initialize().await {
            warn!("Failed to initialize tokenizer manager: {}", e);
            info!("Continuing without tokenizer support");
            None
        } else {
            info!("Tokenizer manager initialized successfully");
            info!("Loaded {} tokenizers", config.tokenizers.len());
            Some(Arc::new(manager))
        }
    } else {
        info!("No tokenizers configured");
        None
    };

    let client = baseten_performance_client_core::PerformanceClientCore::new(
        config
            .default_target_url
            .clone()
            .unwrap_or_else(|| "https://localhost".to_string()),
        Some(
            config
                .upstream_api_key
                .clone()
                .unwrap_or("invalid_key".to_string()),
        ),
        config.http_version,
        None,
        None,
    )
    .map_err(|e| {
        error!("Failed to create performance client: {}", e);
        format!("Failed to create performance client: {}", e)
    })?;

    let handler = UnifiedHandler::new(config.clone(), Arc::new(client), tokenizer_manager);

    // Build the application with CORS and tracing middleware
    let app = Router::new()
        .route("/v1/embeddings", any(handle_unified_request))
        .route("/rerank", any(handle_unified_request))
        .route("/predict", any(handle_unified_request))
        .route("/classify", any(handle_unified_request))
        .route("/health_internal", get(handle_health_check))
        .route("/*path", any(handle_unified_request)) // Catch-all for generic batch
        .layer(
            ServiceBuilder::new().layer(
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods(Any)
                    .allow_headers(Any),
            ),
        )
        .with_state(handler);

    // Create the TCP listener
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", config.port))
        .await
        .map_err(|e| format!("Failed to bind to port {}: {}", config.port, e))?;

    info!("Baseten Reverse Proxy starting on port {}", config.port);
    match &config.default_target_url {
        Some(url) => info!("Default Target URL: {}", url),
        None => info!("No default target URL configured (must be provided per request)"),
    }
    info!("Available endpoints:");
    info!("  POST /v1/embeddings - OpenAI-compatible embeddings");
    info!("  POST /rerank - Reranking service");
    info!("  POST /predict - Classification service");
    info!("  POST /classify - Alternative classification endpoint");
    info!("  GET /health_internal - Internal health check endpoint");
    info!("  ANY /*path - Generic batch requests");

    // Start the server with graceful shutdown
    info!("Server starting up...");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| format!("Server failed: {}", e))?;

    info!("Server shutdown complete");
    Ok(())
}

async fn handle_unified_request(
    State(handler): State<UnifiedHandler>,
    request: Request,
) -> impl IntoResponse {
    let (parts, body) = request.into_parts();
    let path = parts.uri.path();
    let method = convert_method(&parts.method);
    let headers = parts.headers;

    debug!("Received {:?} request to {}", method, path);

    // Read the request body
    let body_bytes = match axum::body::to_bytes(body, constants::MAX_PROXY_SIZE).await {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("Failed to read request body: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Failed to read request body"})),
            )
                .into_response();
        }
    };

    // Parse the request body as JSON
    let body_value: Value = match serde_json::from_slice(&body_bytes) {
        Ok(value) => value,
        Err(e) => {
            warn!("Failed to parse request body as JSON: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Invalid JSON in request body"})),
            )
                .into_response();
        }
    };

    // Process the request through the unified handler
    match handler
        .handle_request(path, method, headers, body_value)
        .await
    {
        Ok((response, response_headers)) => {
            debug!("Successfully processed request to {}", path);
            let mut axum_response = (StatusCode::OK, Json(response)).into_response();
            axum_response.headers_mut().extend(response_headers);
            axum_response
        }
        Err((status, error_message)) => {
            warn!("Request to {} failed: {}", path, error_message);
            (
                status,
                Json(json!({
                    "error": error_message,
                    "path": path,
                    "method": format!("{:?}", method)
                })),
            )
                .into_response()
        }
    }
}

async fn handle_health_check() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(json!({
            "status": "healthy",
            "service": "baseten-performance-proxy",
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        })),
    )
}

// Convert axum::http::Method to our HttpMethod
fn convert_method(method: &Method) -> HttpMethod {
    match method.as_str() {
        "GET" => HttpMethod::GET,
        "POST" => HttpMethod::POST,
        "PUT" => HttpMethod::PUT,
        "PATCH" => HttpMethod::PATCH,
        "DELETE" => HttpMethod::DELETE,
        "HEAD" => HttpMethod::HEAD,
        "OPTIONS" => HttpMethod::OPTIONS,
        _ => HttpMethod::POST, // Default to POST for unknown methods
    }
}

// Handle shutdown signals (Ctrl+C, SIGTERM)
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(unix)]
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    #[cfg(not(unix))]
    tokio::select! {
        _ = ctrl_c => {},
    }

    info!("Received shutdown signal, starting graceful shutdown...");
}

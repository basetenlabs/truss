use axum::{
    extract::{Request, State},
    http::{StatusCode, Method},
    response::{IntoResponse, Json},
    routing::{any, get},
    Router,
};
use baseten_performance_client_core::HttpMethod;
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::{debug, error, info, warn};

use crate::config::ProxyConfig;
use crate::handlers::UnifiedHandler;
use crate::constants;

pub async fn create_server(config: Arc<ProxyConfig>) -> Result<(), Box<dyn std::error::Error>> {
    let handler = UnifiedHandler::new(config.clone());

    // Build the application with CORS and tracing middleware
    let app = Router::new()
        .route("/v1/embeddings", any(handle_unified_request))
        .route("/rerank", any(handle_unified_request))
        .route("/predict", any(handle_unified_request))
        .route("/classify", any(handle_unified_request))
        .route("/health_internal", get(handle_health_check))
        .route("/*path", any(handle_unified_request)) // Catch-all for generic batch
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(
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

    // Start the server
    axum::serve(listener, app)
        .await
        .map_err(|e| format!("Server failed: {}", e))?;

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
    match handler.handle_request(path, method, headers, body_value).await {
        Ok(response) => {
            debug!("Successfully processed request to {}", path);
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(status) => {
            let error_message = match status {
                StatusCode::UNAUTHORIZED => "Unauthorized - missing or invalid API key",
                StatusCode::BAD_REQUEST => "Bad request - invalid parameters or headers",
                StatusCode::BAD_GATEWAY => "Bad gateway - upstream service error",
                StatusCode::INTERNAL_SERVER_ERROR => "Internal server error",
                _ => "Unknown error",
            };

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
            "service": "baseten-reverse-proxy",
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

use crate::errors::ClientError;

use async_trait::async_trait;
use std::fmt;
use std::sync::Arc;

/// Supplies the bearer token for short-lived credentials, such as Baseten access
/// tokens obtained through workload identity federation.
///
/// The client calls `token` before every HTTP attempt, so implementations should
/// cache the token and only fetch a new one when it is close to expiring.
#[async_trait]
pub trait TokenProvider: Send + Sync {
    async fn token(&self) -> Result<String, ClientError>;
}

/// Where the bearer token for a request comes from.
#[derive(Clone)]
pub enum Credential {
    ApiKey(String),
    TokenProvider(Arc<dyn TokenProvider>),
}

impl Credential {
    pub async fn token(&self) -> Result<String, ClientError> {
        match self {
            Credential::ApiKey(key) => Ok(key.clone()),
            Credential::TokenProvider(provider) => {
                let token = provider.token().await?;
                if token.is_empty() {
                    return Err(ClientError::TokenProvider(
                        "token provider returned an empty token".to_string(),
                    ));
                }
                Ok(token)
            }
        }
    }
}

impl fmt::Debug for Credential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Credential::ApiKey(_) => f.write_str("Credential::ApiKey(<redacted>)"),
            Credential::TokenProvider(_) => f.write_str("Credential::TokenProvider"),
        }
    }
}

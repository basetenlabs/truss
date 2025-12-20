use std::collections::HashMap;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Response {
    status_code: u16,
    headers: HashMap<String, String>,
}

impl Response {
    pub fn new(status_code: u16, headers: HashMap<String, String>) -> Self {
        Self {
            status_code,
            headers,
        }
    }

    /// Returns the status code of this response.
    pub fn status(&self) -> u16 {
        self.status_code
    }

    /// Returns the list of header keys present in this response.
    pub fn get_header_keys(&self) -> Vec<&str> {
        self.headers.keys().map(|key| key.as_str()).collect()
    }

    /// Returns the value of a header.
    ///
    /// If the header contains more than one value, only the first value is returned.
    pub fn get_header_value(&self, key: &str) -> Option<&str> {
        self.headers.get(key).map(|s| s.as_str())
    }

    /// Returns all values for a header.
    ///
    /// If the header contains only one value, it will be returned as a single-element vector.
    pub fn get_header_values(&self, key: &str) -> Vec<&str> {
        self.headers
            .get(key)
            .map(|v| vec![v.as_str()])
            .unwrap_or_default()
    }

    /// Returns a copy of all headers
    pub fn headers(&self) -> HashMap<String, String> {
        self.headers.clone()
    }
}

pub struct ErrorBody {
    // For now, we don't need to store the body since we're focused on parsing
    // This is a placeholder for compatibility with the original eventsource client
}

impl ErrorBody {
    pub fn new() -> Self {
        Self {}
    }
}

impl std::fmt::Debug for ErrorBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ErrorBody").finish()
    }
}

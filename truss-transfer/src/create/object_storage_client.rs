use object_store::client::{ClientConfigKey, ClientOptions};

pub fn get_client_options() -> ClientOptions {
    for var in &[
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ] {
        std::env::remove_var(var);
    }
    // In case libraries read NO_PROXY
    std::env::set_var("NO_PROXY", "*");
    std::env::set_var("no_proxy", "*");

    const CLOUD_PROXY_EXCLUDES: &str = concat!(
        ".amazonaws.com,",
        ".s3.amazonaws.com,",
        "s3-accelerate.amazonaws.com,",
        ".googleapis.com,",
        "storage.googleapis.com,",
        ".blob.core.windows.net,",
        ".dfs.core.windows.net,",
        ".core.windows.net,",
        "169.254.169.254"
    );

    ClientOptions::new().with_config(ClientConfigKey::ProxyExcludes, CLOUD_PROXY_EXCLUDES)
}

use object_store::client::{ClientConfigKey, ClientOptions};

pub fn get_client_options() -> ClientOptions {
    ClientOptions::new().with_config(ClientConfigKey::ProxyExcludes, "*")
}

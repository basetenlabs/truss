use baseten_performance_client_core::PerformanceClientCore;

#[test]
fn global_tracing_initialization_follows_feature() {
    assert_eq!(
        PerformanceClientCore::get_api_key(Some("test".into())).unwrap(),
        "test"
    );
    let result = tracing::subscriber::set_global_default(tracing_subscriber::registry());
    assert_eq!(result.is_err(), cfg!(feature = "auto-init-tracing"));
}

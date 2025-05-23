def test_bei_client_bindings_basic_test():
    from bei_client import PerformanceClient

    PerformanceClient.embed
    PerformanceClient.rerank
    PerformanceClient.classify
    PerformanceClient.batch_post

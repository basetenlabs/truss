def test_baseten_performance_client_bindings_basic_test():
    from baseten_performance_client import PerformanceClient

    PerformanceClient.embed
    PerformanceClient.async_embed
    PerformanceClient.rerank
    PerformanceClient.async_rerank
    PerformanceClient.classify
    PerformanceClient.async_classify
    PerformanceClient.batch_post
    PerformanceClient.async_batch_post

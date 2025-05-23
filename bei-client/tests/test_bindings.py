def test_bei_client_bindings_basic_test():
    from bei_client import InferenceClient

    InferenceClient.embed
    InferenceClient.rerank
    InferenceClient.classify
    InferenceClient.batch_post

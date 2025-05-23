def test_baseten_inference_client_bindings_basic_test():
    from baseten_inference_client import InferenceClient

    InferenceClient.embed
    InferenceClient.rerank
    InferenceClient.classify
    InferenceClient.batch_post

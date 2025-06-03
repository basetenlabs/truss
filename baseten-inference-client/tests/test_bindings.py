def test_baseten_inference_client_bindings_basic_test():
    from baseten_inference_client import InferenceClient

    InferenceClient.embed
    InferenceClient.async_embed
    InferenceClient.rerank
    InferenceClient.async_rerank
    InferenceClient.classify
    InferenceClient.async_classify
    InferenceClient.batch_post
    InferenceClient.async_batch_post

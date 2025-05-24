def test_baseten_inference_client_bindings_basic_test():
    from baseten_inference_client import InferenceClient

    InferenceClient.embed
    InferenceClient.aembed
    InferenceClient.rerank
    InferenceClient.arerank
    InferenceClient.classify
    InferenceClient.aclassify
    InferenceClient.batch_post
    InferenceClient.abatch_post

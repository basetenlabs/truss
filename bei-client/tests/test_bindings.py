def test_bei_client_bindings_basic_test():
    from bei_client import SyncClient

    SyncClient.embed
    SyncClient.rerank
    SyncClient.classify

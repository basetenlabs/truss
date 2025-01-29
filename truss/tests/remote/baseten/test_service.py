from truss.remote.baseten import service


def test_model_status_page_url():
    url = service.URLConfig.status_page_url(
        "https://app.baseten.co", service.URLConfig.MODEL, "123"
    )
    assert url == "https://app.baseten.co/models/123/overview"


def test_chain_status_page_url():
    url = service.URLConfig.status_page_url(
        "https://app.baseten.co", service.URLConfig.CHAIN, "abc"
    )
    assert url == "https://app.baseten.co/chains/abc/overview"


def test_model_logs_url():
    url = service.URLConfig.model_logs_url("https://app.baseten.co", "123", "789")
    assert url == "https://app.baseten.co/models/123/logs/789"


def test_chain_logs_url():
    url = service.URLConfig.chainlet_logs_url(
        "https://app.baseten.co", "abc", "666", "543"
    )
    assert url == "https://app.baseten.co/chains/abc/logs/666/543"

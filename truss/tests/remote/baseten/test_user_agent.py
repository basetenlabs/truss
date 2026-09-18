import re

import truss
from truss.remote.baseten import user_agent


def test_user_agent_header_format():
    header = user_agent.user_agent_header()
    assert re.fullmatch(
        rf"truss-(cli|sdk)/{re.escape(truss.__version__)} \(Python/[\d.]+; \w+\)",
        header,
    )


def test_set_client_name_changes_product_token():
    original = user_agent._client_name
    try:
        user_agent.set_client_name("truss-cli")
        assert user_agent.user_agent_header().startswith("truss-cli/")
        user_agent.set_client_name("truss-sdk")
        assert user_agent.user_agent_header().startswith("truss-sdk/")
    finally:
        user_agent.set_client_name(original)


def test_with_user_agent_injects_when_absent():
    out = user_agent.with_user_agent({"Authorization": "Api-Key token"})
    assert out["Authorization"] == "Api-Key token"
    assert out["User-Agent"] == user_agent.user_agent_header()


def test_with_user_agent_does_not_clobber_existing():
    out = user_agent.with_user_agent({"User-Agent": "custom/1.0"})
    assert out == {"User-Agent": "custom/1.0"}


def test_with_user_agent_does_not_clobber_case_insensitive():
    out = user_agent.with_user_agent({"user-agent": "custom/1.0"})
    assert out == {"user-agent": "custom/1.0"}

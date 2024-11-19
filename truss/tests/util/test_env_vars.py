import os

from truss.util.env_vars import override_env_vars


def test_override_env_vars():
    os.environ["API_KEY"] = "original_key"

    with override_env_vars({"API_KEY": "new_key", "DEBUG": "true"}):
        assert os.environ["API_KEY"] == "new_key"
        assert os.environ["DEBUG"] == "true"

    assert os.environ["API_KEY"] == "original_key"
    assert "DEBUG" not in os.environ

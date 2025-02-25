import os

from truss.util.env_vars import override_env_vars


def test_override_env_vars():
    os.environ["API_KEY"] = "original_key"
    os.environ["KEY_OVERRIDDEN_WITH_NONE"] = "original_key"

    with override_env_vars(
        {"API_KEY": "new_key", "DEBUG": "true", "KEY_OVERRIDDEN_WITH_NONE": None}
    ):
        assert os.environ["API_KEY"] == "new_key"
        assert os.environ["DEBUG"] == "true"
        assert "KEY_OVERRIDDEN_WITH_NONE" not in os.environ

    assert os.environ["API_KEY"] == "original_key"
    assert "DEBUG" not in os.environ
    assert os.environ["KEY_OVERRIDDEN_WITH_NONE"] == "original_key"

import os

from truss.util.env_vars import override_env_vars


def test_override_env_vars():
    os.environ["API_KEY"] = "original_key"
    os.environ["AWS_CONFIG_FILE"] = "original_config_file"

    with override_env_vars(
        env_vars={"API_KEY": "new_key", "DEBUG": "true"},
        deleted_vars={"AWS_CONFIG_FILE"},
    ):
        assert os.environ["API_KEY"] == "new_key"
        assert os.environ["DEBUG"] == "true"
        assert "AWS_CONFIG_FILE" not in os.environ

    assert os.environ["API_KEY"] == "original_key"
    assert "DEBUG" not in os.environ
    assert os.environ["AWS_CONFIG_FILE"] == "original_config_file"

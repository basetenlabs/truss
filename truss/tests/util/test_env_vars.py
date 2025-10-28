import os

from truss.util.env_vars import modify_env_vars


def test_modify_env_vars():
    os.environ["API_KEY"] = "original_key"
    os.environ["AWS_CONFIG_FILE"] = "original_config_file"

    with modify_env_vars(
        overrides={"API_KEY": "new_key", "DEBUG": "true"}, deletions={"AWS_CONFIG_FILE"}
    ):
        assert os.environ["API_KEY"] == "new_key"
        assert os.environ["DEBUG"] == "true"
        assert "AWS_CONFIG_FILE" not in os.environ

    assert os.environ["API_KEY"] == "original_key"
    assert "DEBUG" not in os.environ
    assert os.environ["AWS_CONFIG_FILE"] == "original_config_file"

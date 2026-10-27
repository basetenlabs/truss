from truss.base.truss_spec import TrussSpec
from truss.contexts.local_loader.utils import prepare_secrets
from truss.local.local_config_handler import LocalConfigHandler
from truss.truss_handle.truss_handle import TrussHandle


def test_prepare_secrets(custom_model_truss_dir, tmp_path):
    orig_truss_config_dir = LocalConfigHandler.TRUSS_CONFIG_DIR
    LocalConfigHandler.TRUSS_CONFIG_DIR = tmp_path
    try:
        LocalConfigHandler.set_secret("secret_name", "secret_value")
        handle = TrussHandle(custom_model_truss_dir)
        handle.add_secret("secret_name")
        spec = TrussSpec(custom_model_truss_dir)
        secrets = prepare_secrets(spec)
        assert secrets["secret_name"] == "secret_value"
    finally:
        LocalConfigHandler.TRUSS_CONFIG_DIR = orig_truss_config_dir


def test_prepare_secrets_list(custom_model_truss_dir, tmp_path):
    orig_truss_config_dir = LocalConfigHandler.TRUSS_CONFIG_DIR
    LocalConfigHandler.TRUSS_CONFIG_DIR = tmp_path
    try:
        LocalConfigHandler.set_secret("secret_name", "secret_value")
        handle = TrussHandle(custom_model_truss_dir)
        handle._update_config(secrets=["secret_name", "other_name"])
        spec = TrussSpec(custom_model_truss_dir)
        secrets = prepare_secrets(spec)
        assert secrets == {"secret_name": "secret_value", "other_name": None}
    finally:
        LocalConfigHandler.TRUSS_CONFIG_DIR = orig_truss_config_dir


def test_add_secret_list(custom_model_truss_dir):
    handle = TrussHandle(custom_model_truss_dir)
    handle._update_config(secrets=["existing"])
    handle.add_secret("new")
    handle.add_secret("new")
    assert TrussSpec(custom_model_truss_dir).config.secrets == ["existing", "new"]
    # A default value can't be expressed in list form.
    handle.add_secret("with_value", "value")
    assert TrussSpec(custom_model_truss_dir).config.secrets == {
        "existing": None,
        "new": None,
        "with_value": "value",
    }

from truss.contexts.local_loader.load_local import _prepare_secrets
from truss.local.local_config_handler import LocalConfigHandler
from truss.truss_handle import TrussHandle
from truss.truss_spec import TrussSpec


def test_prepare_secrets(custom_model_truss_dir, tmp_path):
    orig_truss_config_dir = LocalConfigHandler.TRUSS_CONFIG_DIR
    LocalConfigHandler.TRUSS_CONFIG_DIR = tmp_path
    try:
        LocalConfigHandler.set_secret('secret_name', 'secret_value')
        handle = TrussHandle(custom_model_truss_dir)
        handle.add_secret('secret_name')
        spec = TrussSpec(custom_model_truss_dir)
        secrets = _prepare_secrets(spec)
        assert secrets['secret_name'] == 'secret_value'
    finally:
        LocalConfigHandler.TRUSS_CONFIG_DIR = orig_truss_config_dir

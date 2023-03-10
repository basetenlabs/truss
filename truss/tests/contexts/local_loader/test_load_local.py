from unittest.mock import patch

from truss.contexts.local_loader.utils import prepare_secrets
from truss.local.local_config_handler import LocalConfigHandler
from truss.truss_handle import TrussHandle
from truss.truss_spec import TrussSpec


def test_prepare_secrets(custom_model_truss_dir, tmp_path):
    with patch("truss.local.local_config_handler.DOT_TRUSS_DIR", tmp_path):
        LocalConfigHandler.set_secret("secret_name", "secret_value")
        handle = TrussHandle(custom_model_truss_dir)
        handle.add_secret("secret_name")
        spec = TrussSpec(custom_model_truss_dir)
        secrets = prepare_secrets(spec)
        assert secrets["secret_name"] == "secret_value"

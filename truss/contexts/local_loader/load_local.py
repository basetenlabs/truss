
import copy
import inspect
from pathlib import Path
from typing import Dict

from truss.contexts.local_loader.model_module_loader import \
    model_class_module_loaded
from truss.contexts.truss_context import TrussContext
from truss.local.local_config_handler import LocalConfigHandler
from truss.truss_spec import TrussSpec


class LoadLocal(TrussContext):
    """Loads a Truss model locally.

    The loaded model can be used to make predictions for quick testing.
    Runs in the current pip environment directly. Assumes all requirements and
    system packages are already installed.
    """

    @staticmethod
    def run(truss_dir: Path):
        spec = TrussSpec(truss_dir)
        with model_class_module_loaded(str(truss_dir), spec.model_module_fullname) as module:
            model_class = getattr(module, spec.model_class_name)
            model_class_signature = inspect.signature(model_class)
            model_init_params = {}
            if _signature_accepts_keyword_arg(model_class_signature, 'config'):
                model_init_params['config'] = spec.config.to_dict()
            if _signature_accepts_keyword_arg(model_class_signature, 'data_dir'):
                model_init_params['data_dir'] = truss_dir / 'data'
            if _signature_accepts_keyword_arg(model_class_signature, 'secrets'):
                model_init_params['secrets'] = _prepare_secrets(spec)
            model = model_class(**model_init_params)
            if hasattr(model, 'load'):
                model.load()
            return model


def _signature_accepts_keyword_arg(signature: inspect.Signature, kwarg: str) -> bool:
    return kwarg in signature.parameters or _signature_accepts_kwargs(signature)


def _signature_accepts_kwargs(signature: inspect.Signature) -> bool:
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _prepare_secrets(spec: TrussSpec) -> Dict[str, str]:
    secrets = copy.deepcopy(spec.secrets)
    local_secerts = LocalConfigHandler.get_config().secrets
    for secret_name in secrets:
        if secret_name in local_secerts:
            secrets[secret_name] = local_secerts[secret_name]
    return secrets

import configparser
import inspect
from pathlib import Path
from typing import Dict, Type

from truss.remote.baseten import BasetenRemote
from truss.remote.truss_remote import TrussRemote


class RemoteFactory:
    """
    A factory for instantiating a TrussRemote from a .trussrc file and a user-specified remote config name
    """

    REGISTRY: Dict[str, Type[TrussRemote]] = {"baseten": BasetenRemote}

    @staticmethod
    def load_remote_config(remote_name: str) -> Dict:
        """
        Load and validate a remote config from the .trussrc file
        """
        config_path = Path("~/.trussrc").expanduser()

        if not config_path.exists():
            raise FileNotFoundError(f"No .trussrc file found at {config_path}")

        config = configparser.ConfigParser()
        config.read(config_path)

        if remote_name not in config:
            raise ValueError(f"Service provider {remote_name} not found in .trussrc")

        return dict(config[remote_name])

    @staticmethod
    def validate_remote_config(remote_config: Dict, remote_name: str):
        """
        Validates remote config by checking
            1. the 'remote' field exists
            2. all required parameters for the 'remote' class are provided
        """
        if "remote_provider" not in remote_config:
            raise ValueError(
                f"Missing 'remote_provider' field for remote {remote_name} in .trussrc"
            )

        if remote_config["remote_provider"] not in RemoteFactory.REGISTRY:
            raise ValueError(
                f"Remote provider {remote_config['remote_provider']} not found in registry"
            )

        remote = RemoteFactory.REGISTRY.get(remote_config["remote_provider"])
        if remote:
            required_params = RemoteFactory.required_params(remote)
            missing_params = required_params - set(remote_config.keys())
            if missing_params:
                raise ValueError(
                    f"Missing required parameter(s) {missing_params} for remote {remote_name} in .trussrc"
                )

    @staticmethod
    def required_params(remote: Type[TrussRemote]) -> set:
        """
        Get the required parameters for a remote by inspecting its __init__ method
        """
        init_signature = inspect.signature(remote.__init__)
        params = init_signature.parameters
        required_params = {
            name
            for name, param in params.items()
            if param.default == inspect.Parameter.empty
            and name not in {"self", "args", "kwargs"}
        }
        return required_params

    @classmethod
    def create(cls, remote: str) -> TrussRemote:
        remote_config = cls.load_remote_config(remote)
        cls.validate_remote_config(remote_config, remote)

        remote_class = cls.REGISTRY[remote_config.pop("remote_provider")]
        remote_params = {
            param: remote_config.get(param)
            for param in cls.required_params(remote_class)
        }

        # Add any additional params provided by the user in their .trussrc
        additional_params = set(remote_config.keys()) - set(remote_params.keys())
        for param in additional_params:
            remote_params[param] = remote_config.get(param)

        return remote_class(**remote_params)  # type: ignore

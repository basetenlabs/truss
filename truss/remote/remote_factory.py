import configparser
import inspect
from pathlib import Path
from typing import Dict, Type

from truss.remote.baseten import BasetenRemote
from truss.remote.truss_remote import TrussRemote


class RemoteFactory:
    """
    A factory for instantiating a TrussRemote from a .trussrc file and a user-specified service name
    """

    REGISTRY: Dict[str, Type[TrussRemote]] = {"baseten": BasetenRemote}

    @staticmethod
    def load_service(remote_name: str) -> Dict:
        """
        Load and validate a service from the .trussrc file
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
    def validate_service(service: Dict, remote_name: str):
        """
        Validates service by checking the 'remote' field and the required parameters
        """
        if "remote_provider" not in service:
            raise ValueError(
                f"Missing 'remote_provider' field for remote {remote_name} in .trussrc"
            )

        if service["remote_provider"] not in RemoteFactory.REGISTRY:
            raise ValueError(
                f"Remote provider {service['remote_provider']} not found in registry"
            )

        remote = RemoteFactory.REGISTRY.get(service["remote_provider"])
        if remote:
            required_params = RemoteFactory.required_params(remote)
            missing_params = required_params - set(service.keys())
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
        service = cls.load_service(remote)
        cls.validate_service(service, remote)

        remote_class = cls.REGISTRY[service.pop("remote_provider")]
        remote_params = {
            param: service.get(param) for param in cls.required_params(remote_class)
        }

        # Add any additional params provided by the user in their .trussrc
        additional_params = set(service.keys()) - set(remote_params.keys())
        for param in additional_params:
            remote_params[param] = service.get(param)

        return remote_class(**remote_params)  # type: ignore

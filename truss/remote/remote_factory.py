import inspect

try:
    from configparser import DEFAULTSECT
    from configparser import SafeConfigParser as ConfigParser
except ImportError:
    # We need to do this for py312 and onwards.
    from configparser import DEFAULTSECT, ConfigParser  # type: ignore

from functools import partial
from operator import is_not
from pathlib import Path
from typing import Dict, List, Type

from truss.remote.baseten import BasetenRemote
from truss.remote.truss_remote import RemoteConfig, TrussRemote

USER_TRUSSRC_PATH = Path("~/.trussrc").expanduser()


def load_config() -> ConfigParser:
    config = ConfigParser()
    config.read(USER_TRUSSRC_PATH)
    return config


def update_config(config: ConfigParser):
    with open(USER_TRUSSRC_PATH, "w") as configfile:
        config.write(configfile)


class RemoteFactory:
    """
    A factory for instantiating a TrussRemote from a .trussrc file and a user-specified remote config name
    """

    REGISTRY: Dict[str, Type[TrussRemote]] = {"baseten": BasetenRemote}

    @staticmethod
    def get_available_config_names() -> List[str]:
        if not USER_TRUSSRC_PATH.exists():
            return []

        config = load_config()
        return list(filter(partial(is_not, DEFAULTSECT), config.keys()))

    @staticmethod
    def load_remote_config(remote_name: str) -> RemoteConfig:
        """
        Load and validate a remote config from the .trussrc file
        """
        if not USER_TRUSSRC_PATH.exists():
            raise FileNotFoundError("No ~/.trussrc file found.")

        config = load_config()

        if remote_name not in config:
            raise ValueError(f"Service provider {remote_name} not found in ~/.trussrc")

        return RemoteConfig(name=remote_name, configs=dict(config[remote_name]))

    @staticmethod
    def update_remote_config(remote_config: RemoteConfig):
        """
        Load and validate a remote config from the .trussrc file
        """
        config = load_config()
        config[remote_config.name] = remote_config.configs
        update_config(config)

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
        configs = remote_config.configs
        cls.validate_remote_config(configs, remote)

        remote_class = cls.REGISTRY[configs.pop("remote_provider")]
        remote_params = {
            param: configs.get(param) for param in cls.required_params(remote_class)
        }

        # Add any additional params provided by the user in their .trussrc
        additional_params = set(configs.keys()) - set(remote_params.keys())
        for param in additional_params:
            remote_params[param] = configs.get(param)

        return remote_class(**remote_params)  # type: ignore

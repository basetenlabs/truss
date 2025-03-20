import inspect
import os

try:
    from configparser import DEFAULTSECT, ConfigParser  # type: ignore
except ImportError:
    # We need to do this for old python.
    from configparser import DEFAULTSECT
    from configparser import SafeConfigParser as ConfigParser


from functools import partial
from operator import is_not
from pathlib import Path
from typing import Dict, List, Type

from truss.remote.baseten import BasetenRemote
from truss.remote.truss_remote import RemoteConfig, TrussRemote

USER_TRUSSRC_PATH = Path(os.environ.get("USER_TRUSSRC_PATH", "~/.trussrc")).expanduser()


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

    @classmethod
    def create(cls, remote: str) -> TrussRemote:
        remote_config = cls.load_remote_config(remote).configs
        if "remote_provider" not in remote_config:
            raise ValueError(f"Missing 'remote_provider' field for remote `{remote}`.")
        provider = remote_config.pop("remote_provider")
        if provider not in cls.REGISTRY:
            raise ValueError(f"Remote provider {provider} not found in registry.")
        remote_class = cls.REGISTRY[provider]

        parameters = inspect.signature(remote_class.__init__).parameters
        init_params = {n for n in parameters if n not in {"self", "args", "kwargs"}}
        required_params = {
            n
            for n, p in parameters.items()
            if p.default == inspect.Parameter.empty
            and n not in {"self", "args", "kwargs"}
        }
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )

        # If class accepts **kwargs, pass everything, else only known params
        if accepts_kwargs:
            passed_config = remote_config
        else:
            passed_config = {k: v for k, v in remote_config.items() if k in init_params}

        missing = required_params - set(remote_config.keys())
        if missing:
            raise ValueError(
                f"Missing required parameter(s) {list(missing)} for remote `{remote}`."
            )
        return remote_class(**passed_config)

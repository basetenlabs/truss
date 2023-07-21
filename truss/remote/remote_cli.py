from typing import Any, Iterable, List, Mapping
from urllib.parse import urlparse

import questionary
import rich
from questionary import Separator, prompt
from truss.remote.remote_factory import RemoteFactory
from truss.remote.truss_remote import RemoteConfig


def uri_validator(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def inquire_remote_config() -> RemoteConfig:
    # TODO(bola): extract questions from remote
    rich.print("💻 Let's add a Baseten remote!")
    questions: Iterable[Mapping[str, Any]] = [
        {
            "type": "text",
            "name": "name",
            "message": "📝 Name to reference this account:",
        },
        {
            "type": "confirm",
            "name": "on_prem",
            "message": "Are you using self-hosting Baseten?",
            "default": False,
        },
        {
            "type": "text",
            "name": "remote_url",
            "message": "🌐 What's the url to reach it?",
            "when": lambda x: x["on_prem"],
            "validate": lambda val: uri_validator(val),
        },
        {
            "type": "password",
            "name": "api_key",
            "message": "🤫 Quiety paste your API_KEY:",
        },
    ]
    answers = prompt(
        questions, {"on_prem": False, "remote_url": "https://app.baseten.co"}
    )
    answers["remote_provider"] = "baseten"
    answers.pop("on_prem")
    remote_name = answers.pop("name")
    return RemoteConfig(name=remote_name, configs=answers)


def inquire_remote_name(available_remotes: List[str]) -> str:
    ADD_REMOTE_QUESTION = "Add new remote"
    if len(available_remotes) > 0:
        remote = questionary.select(
            "Which remote do you want to push to?",
            choices=[*available_remotes, Separator(), ADD_REMOTE_QUESTION],
        ).ask()
        if remote is not ADD_REMOTE_QUESTION:
            return remote
    remote_config = inquire_remote_config()
    RemoteFactory.update_remote_config(remote_config)
    return remote_config.name


def inquire_model_name() -> str:
    return questionary.text("📦 Name this model: ").ask()

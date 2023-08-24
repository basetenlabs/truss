from typing import List

import rich
from InquirerPy import inquirer
from truss.remote.remote_factory import USER_TRUSSRC_PATH, RemoteFactory
from truss.remote.truss_remote import RemoteConfig


def inquire_remote_config() -> RemoteConfig:
    # TODO(bola): extract questions from remote
    rich.print("💻 Let's add a Baseten remote!")
    # If users need to adjust the remote url, they
    # can do so manually in the .trussrc file.
    remote_url = "https://app.baseten.co"
    api_key = inquirer.secret(
        message="🤫 Quietly paste your API_KEY:",
        qmark="",
    ).execute()

    return RemoteConfig(
        name="baseten",
        configs={
            "remote_provider": "baseten",
            "api_key": api_key,
            "remote_url": remote_url,
        },
    )


def inquire_remote_name(available_remotes: List[str]) -> str:
    if len(available_remotes) > 1:
        remote = inquirer.select(
            "🎮 Which remote do you want to connect to?",
            qmark="",
            choices=available_remotes,
        ).execute()
        return remote
    elif len(available_remotes) == 1:
        return available_remotes[0]
    remote_config = inquire_remote_config()
    RemoteFactory.update_remote_config(remote_config)

    rich.print(f"💾 Remote config saved to {USER_TRUSSRC_PATH}")
    return remote_config.name


def inquire_model_name() -> str:
    return inquirer.text(
        "📦 Name this model:",
        qmark="",
    ).execute()

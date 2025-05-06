"""Tools to configure and track the Truss CLI state and behavior, e.g. auto-upgrades."""

import datetime
import logging
import os
import pathlib
import shutil
import subprocess
from typing import Any, Optional, Union

import packaging.version
import pydantic
import requests
import tomlkit
import tomlkit.container
import tomlkit.items

from truss.remote.remote_factory import load_config

_SETTINGS_FiLE = "settings.toml"
_STATE_FILE = "state.json"
_PYPI_VERSION_URL = "https://pypi.org/pypi/truss/json"


def _get_dir() -> pathlib.Path:
    if "XDG_CONFIG_HOME" in os.environ:
        base_dir = pathlib.Path(os.environ["XDG_CONFIG_HOME"])
    else:
        base_dir = pathlib.Path.home() / ".config"

    settings_path = base_dir / "truss"
    settings_path.mkdir(parents=True, exist_ok=True)
    return settings_path


def _truss_is_git_branch() -> bool:
    if shutil.which("git") is None:
        return False
    source_path = pathlib.Path(__file__).resolve()
    source_dir = source_path.parent
    try:
        result = subprocess.run(
            "git rev-parse --is-inside-work-tree && git remote get-url origin",
            cwd=source_dir,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        remote_url = result.stdout.strip().splitlines()[-1]
        remote_matches = "basetenlabs/truss" in remote_url
        if not remote_matches:
            logging.debug(f"Inside truss git repo with remote {remote_url}.")
        return remote_matches
    except subprocess.CalledProcessError:
        return False


class Preferences(pydantic.BaseModel):
    include_git_info: bool = False
    auto_upgrade_command_template: Optional[str] = None


class FeatureFlags(pydantic.BaseModel):
    enable_auto_upgrade: bool = False


class AppSettings(pydantic.BaseModel):
    preferences: Preferences = Preferences()
    feature_flags: FeatureFlags = FeatureFlags()


def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
    return {
        k: _strip_none(v) if isinstance(v, dict) else v
        for k, v in d.items()
        if v is not None
    }


def _update_toml_document(doc, data: dict[str, Any]) -> None:
    """Update the values, but keep structure (esp. comments)."""
    # NOTE: this does not remove keys that are not in `data` from the toml doc.
    data = _strip_none(data)
    for key, value in data.items():
        if isinstance(value, dict) and key in doc:
            _update_toml_document(doc[key], value)
        else:
            if value is None:
                if key in doc:
                    del doc[key]
            else:
                doc[key] = value


class _SettingsWrapper:
    def __init__(
        self, settings: AppSettings, toml_doc: tomlkit.TOMLDocument, write: bool = False
    ):
        self._settings = settings
        self._toml_doc = toml_doc

        if write:
            self._write()

    @staticmethod
    def path() -> pathlib.Path:
        return _get_dir() / _SETTINGS_FiLE

    @classmethod
    def read_or_create(cls) -> "_SettingsWrapper":
        if cls.path().exists():
            toml_doc = tomlkit.parse(cls.path().read_text(encoding="utf-8"))
            settings = AppSettings(**toml_doc.unwrap())
            write = False
        else:
            settings = AppSettings()
            truss_rc = load_config()
            has_git_info_consent = any(
                truss_rc.get(section, "include_git_info", fallback="false")
                .strip()
                .lower()
                == "true"
                for section in truss_rc.sections()
            )
            if has_git_info_consent:
                settings.preferences.include_git_info = True
            toml_doc = tomlkit.document()
            write = True

        return cls(settings, toml_doc, write)

    def _write(self) -> None:
        # The contract is that all manipulations are done on `settings`, not on `toml_doc`.
        _update_toml_document(self._toml_doc, self._settings.model_dump())
        self.path().write_text(tomlkit.dumps(self._toml_doc), encoding="utf-8")

    @property
    def include_git_info(self) -> bool:
        return self._settings.preferences.include_git_info

    @property
    def enable_auto_upgrade(self) -> bool:
        return self._settings.feature_flags.enable_auto_upgrade

    @property
    def auto_upgrade_command_template(self) -> Optional[str]:
        return self._settings.preferences.auto_upgrade_command_template

    @auto_upgrade_command_template.setter
    def auto_upgrade_command_template(self, cmd: Optional[str]) -> None:
        self._settings.preferences.auto_upgrade_command_template = cmd
        self._write()


class VersionInfo(pydantic.BaseModel):
    latest_version: Optional[packaging.version.Version] = None
    yanked_versions: set[packaging.version.Version] = pydantic.Field(
        default_factory=set
    )
    last_check: datetime.datetime = datetime.datetime.min

    @pydantic.field_validator("latest_version", mode="before")
    @classmethod
    def _parse_latest(
        cls, value: Optional[Union[str, packaging.version.Version]]
    ) -> Optional[packaging.version.Version]:
        if value is None or isinstance(value, packaging.version.Version):
            return value
        return packaging.version.Version(value)

    @pydantic.field_validator("yanked_versions", mode="before")
    @classmethod
    def _parse_yanked(
        cls, value: list[Union[str, packaging.version.Version]]
    ) -> list[packaging.version.Version]:
        return [
            v
            if isinstance(v, packaging.version.Version)
            else packaging.version.Version(v)
            for v in value
        ]

    model_config = {
        "json_encoders": {packaging.version.Version: lambda v: str(v)},
        "arbitrary_types_allowed": True,
    }


class State(pydantic.BaseModel):
    version_info: VersionInfo = VersionInfo()


class UpdateInfo(pydantic.BaseModel):
    upgrade_recommended: bool
    reason: str
    latest_version: str


class _StateWrapper:
    def __init__(self, state: State, write: bool = False):
        self._state = state
        if write:
            self._write()

    @staticmethod
    def _path() -> pathlib.Path:
        return _get_dir() / _STATE_FILE

    @classmethod
    def read_or_create(cls):
        if cls._path().exists():
            state = State.model_validate_json(
                cls._path().read_text(encoding="utf-8") or "{}"
            )
            write = False
        else:
            state = State()
            write = True

        return cls(state, write)

    def _write(self) -> None:
        self._path().write_text(self._state.model_dump_json(indent=2), encoding="utf-8")

    def _should_check_for_updates(self) -> bool:
        return (
            self._state.version_info.last_check
            < datetime.datetime.now() - datetime.timedelta(days=1)
        )

    def _update_version_info(self) -> None:
        response = requests.get(_PYPI_VERSION_URL, timeout=5)
        response.raise_for_status()
        data = response.json()

        latest_version = packaging.version.Version(data["info"]["version"])
        assert latest_version, data["info"]["version"]
        yanked_versions = [
            packaging.version.Version(version)
            for version, files in data["releases"].items()
            if any(file.get("yanked", False) for file in files)
        ]
        self._state.version_info = VersionInfo(
            latest_version=latest_version,
            yanked_versions=set(yanked_versions),
            last_check=datetime.datetime.now(),
        )
        self._write()

    def should_upgrade(self, current_version: str) -> UpdateInfo:
        local_version = packaging.version.Version(current_version)
        upgrade_recommended = False
        reason = "Up to date."

        if self._should_check_for_updates():
            self._update_version_info()

        latest_version = self._state.version_info.latest_version
        assert latest_version

        # Note: these conditions are *supposed* to overwrite previous ones.
        if local_version < latest_version:
            reason = f"🐌 The current version '{local_version}' is outdated."
            upgrade_recommended = True

        if local_version.is_devrelease or local_version.is_prerelease:
            reason = "Local version is for dev - upgrades are not applied."
            upgrade_recommended = False

        if local_version in self._state.version_info.yanked_versions:
            reason = f"🧨 The current version '{local_version}' is yanked ."
            upgrade_recommended = True

        if upgrade_recommended and _truss_is_git_branch():
            upgrade_recommended = False
            reason = "Truss is in a git branch, upgrades are not applied."

        return UpdateInfo(
            upgrade_recommended=upgrade_recommended,
            reason=reason,
            latest_version=str(latest_version),
        )


state = _StateWrapper.read_or_create()
settings = _SettingsWrapper.read_or_create()

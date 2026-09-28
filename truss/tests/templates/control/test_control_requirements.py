from packaging.requirements import Requirement
from packaging.version import Version

import truss
from truss.base import constants

CONTROL_REQUIREMENTS = constants.CONTROL_SERVER_CODE_DIR / "requirements.txt"


def _pinned_truss_version() -> Version:
    for line in CONTROL_REQUIREMENTS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        req = Requirement(line)
        if req.name == "truss":
            (spec,) = req.specifier
            assert spec.operator == "==", f"Expected exact truss pin, got `{req}`."
            return Version(spec.version)
    raise AssertionError(f"No truss pin found in {CONTROL_REQUIREMENTS}.")


def test_control_server_truss_pin_tracks_current_minor_version():
    # The control server parses `config.yaml` with its own pinned truss. If the pin
    # falls behind, newer config values (e.g. accelerators) fail validation at startup
    # and development deployments crash.
    current = Version(truss.__version__)
    pinned = _pinned_truss_version()
    assert (pinned.major, pinned.minor) == (current.major, current.minor), (
        f"Control server pins truss=={pinned}, but truss is at {current}. "
        f"Update {CONTROL_REQUIREMENTS} to the latest release."
    )

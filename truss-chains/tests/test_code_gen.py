import os
import stat
import sys
from types import SimpleNamespace

import pytest

from truss_chains import private_types, public_types
from truss_chains.deployment import code_gen

_DESCRIPTOR = SimpleNamespace(name="MyChainlet")


def test_make_chainlet_dir_creates_owner_only_gen_dir(tmp_path):
    chainlet_dir = code_gen._make_chainlet_dir("My Chain", _DESCRIPTOR, tmp_path)

    gen_dir = tmp_path / private_types.GENERATED_CODE_DIR
    assert chainlet_dir == gen_dir / "My_Chain" / "chainlet_MyChainlet"
    assert chainlet_dir.is_dir()
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(gen_dir).st_mode) == 0o700


def test_make_chainlet_dir_reuses_and_tightens_existing_gen_dir(tmp_path):
    gen_dir = tmp_path / private_types.GENERATED_CODE_DIR
    gen_dir.mkdir(mode=0o755)
    stale = gen_dir / "My_Chain" / "chainlet_MyChainlet" / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("old")

    chainlet_dir = code_gen._make_chainlet_dir("My Chain", _DESCRIPTOR, tmp_path)

    assert chainlet_dir.is_dir()
    assert not stale.exists()
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(gen_dir).st_mode) == 0o700


def test_make_chainlet_dir_rejects_non_dir_gen_path(tmp_path):
    (tmp_path / private_types.GENERATED_CODE_DIR).write_text("not a dir")

    with pytest.raises(public_types.ChainsUsageError, match="not a directory owned"):
        code_gen._make_chainlet_dir("My Chain", _DESCRIPTOR, tmp_path)


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_make_chainlet_dir_rejects_symlinked_gen_dir(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    (tmp_path / private_types.GENERATED_CODE_DIR).symlink_to(target)

    with pytest.raises(public_types.ChainsUsageError, match="not a directory owned"):
        code_gen._make_chainlet_dir("My Chain", _DESCRIPTOR, tmp_path)
    assert list(target.iterdir()) == []


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="no POSIX uids")
def test_make_chainlet_dir_rejects_gen_dir_owned_by_other_user(tmp_path, monkeypatch):
    (tmp_path / private_types.GENERATED_CODE_DIR).mkdir()
    other_uid = os.getuid() + 1
    monkeypatch.setattr(os, "getuid", lambda: other_uid)

    with pytest.raises(public_types.ChainsUsageError, match="not a directory owned"):
        code_gen._make_chainlet_dir("My Chain", _DESCRIPTOR, tmp_path)

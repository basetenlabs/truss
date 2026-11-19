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


def test_make_chainlet_dir_reuses_and_tightens_existing_dirs(tmp_path):
    gen_dir = tmp_path / private_types.GENERATED_CODE_DIR
    chain_dir = gen_dir / "My_Chain"
    old_chainlet_dir = chain_dir / "chainlet_MyChainlet"
    for path in (gen_dir, chain_dir, old_chainlet_dir):
        path.mkdir(mode=0o755)
    stale = old_chainlet_dir / "stale.txt"
    stale.write_text("old")

    chainlet_dir = code_gen._make_chainlet_dir("My Chain", _DESCRIPTOR, tmp_path)

    assert chainlet_dir == old_chainlet_dir
    assert chainlet_dir.is_dir()
    assert not stale.exists()
    if sys.platform != "win32":
        for path in (gen_dir, chain_dir, chainlet_dir):
            assert stat.S_IMODE(os.stat(path).st_mode) == 0o700


@pytest.mark.parametrize("chain_name", [".", "..", " .. "])
def test_make_chainlet_dir_rejects_path_special_chain_names(tmp_path, chain_name):
    root_mode = os.stat(tmp_path).st_mode

    with pytest.raises(public_types.ChainsUsageError, match="cannot be used"):
        code_gen._make_chainlet_dir(chain_name, _DESCRIPTOR, tmp_path)
    assert os.stat(tmp_path).st_mode == root_mode


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


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
@pytest.mark.parametrize(
    "link_name, target_name",
    [
        ("My_Chain", "outside"),
        ("My_Chain/chainlet_MyChainlet", "outside/chainlet_MyChainlet"),
    ],
)
def test_make_chainlet_dir_rejects_symlink_below_gen_dir(
    tmp_path, link_name, target_name
):
    marker = tmp_path / "outside" / "chainlet_MyChainlet" / "marker"
    marker.parent.mkdir(parents=True)
    marker.write_text("keep")
    link = tmp_path / private_types.GENERATED_CODE_DIR / link_name
    link.parent.mkdir(parents=True)
    link.symlink_to(tmp_path / target_name)

    with pytest.raises(public_types.ChainsUsageError, match="not a directory owned"):
        code_gen._make_chainlet_dir("My Chain", _DESCRIPTOR, tmp_path)
    assert marker.read_text() == "keep"


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="no POSIX uids")
@pytest.mark.parametrize(
    "foreign_name",
    [private_types.GENERATED_CODE_DIR, "My_Chain", "chainlet_MyChainlet"],
)
def test_make_chainlet_dir_rejects_dir_owned_by_other_user(
    tmp_path, monkeypatch, foreign_name
):
    real_lstat = os.lstat

    def lstat_with_foreign_owner(path):
        st = real_lstat(path)
        if os.path.basename(path) != foreign_name:
            return st
        return SimpleNamespace(st_mode=st.st_mode, st_uid=st.st_uid + 1)

    monkeypatch.setattr(os, "lstat", lstat_with_foreign_owner)

    with pytest.raises(public_types.ChainsUsageError, match="not a directory owned"):
        code_gen._make_chainlet_dir("My Chain", _DESCRIPTOR, tmp_path)

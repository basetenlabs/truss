import random
import string
from pathlib import Path
from typing import Callable, List

import pytest
from truss.patch.hash import (
    directory_content_hash,
    file_content_hash,
    file_content_hash_str,
)


@pytest.fixture
def dir_hash_test_dir(tmp_path: Path):
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target_dir_file = target_dir / "target_file"
    _update_file_content(target_dir_file)
    sub_dir = target_dir / "subdir"
    sub_dir.mkdir()
    sub_dir_file = sub_dir / "subdir_file"
    _update_file_content(sub_dir_file)
    return target_dir


def test_dir_hash_same_on_retry(dir_hash_test_dir):
    _verify_with_dir_modification(dir_hash_test_dir, lambda _: None, True)


def test_dir_hash_different_if_file_added_at_top_level(dir_hash_test_dir):
    def mod(dir_path):
        _update_file_content(dir_path / "new_file")

    _verify_with_dir_modification(dir_hash_test_dir, mod, False)


def test_dir_hash_different_if_file_added_to_sub_directory(dir_hash_test_dir):
    def mod(dir_path):
        _update_file_content(dir_path / "subdir" / "new_file")

    _verify_with_dir_modification(dir_hash_test_dir, mod, False)


def test_dir_hash_different_if_file_file_content_modified(dir_hash_test_dir):
    def mod(dir_path):
        _update_file_content(
            dir_path / "target_file", content=_generate_random_string(5)
        )

    _verify_with_dir_modification(dir_hash_test_dir, mod, False)


def test_dir_hash_same_if_target_dir_renamed(tmp_path, dir_hash_test_dir):
    def mod(dir_path):
        target = tmp_path / "renamed"
        dir_path.rename(target)
        return target

    _verify_with_dir_modification(dir_hash_test_dir, mod, True)


def test_dir_hash_same_if_target_dir_moved_but_not_renamed(tmp_path, dir_hash_test_dir):
    def mod(dir_path):
        holding_dir = tmp_path / "holding_dir"
        holding_dir.mkdir()
        target = holding_dir / dir_path.name
        dir_path.rename(target)
        return target

    _verify_with_dir_modification(dir_hash_test_dir, mod, True)


def test_dir_hash_different_if_sub_directory_renamed(dir_hash_test_dir):
    def mod(dir_path):
        sub_dir = dir_path / "subdir"
        sub_dir.rename(dir_path / "new_subdir")

    _verify_with_dir_modification(dir_hash_test_dir, mod, False)


def test_dir_hash_different_if_file_contents_swapped(dir_hash_test_dir):
    _update_file_content(dir_hash_test_dir / "file1")
    _update_file_content(dir_hash_test_dir / "file2")

    def mod(dir_path):
        tmp_file = dir_path / "tmp_file"
        (dir_path / "file1").rename(tmp_file)
        (dir_path / "file2").rename(dir_path / "file1")
        tmp_file.rename(dir_path / "file2")

    _verify_with_dir_modification(dir_hash_test_dir, mod, False)


def test_dir_hash_ignore_pattern_filename(dir_hash_test_dir):
    _update_file_content(dir_hash_test_dir / "file1")
    _update_file_content(dir_hash_test_dir / "file2")

    def mod_file1(dir_path):
        _update_file_content(dir_path / "file1")

    def mod_file2(dir_path):
        _update_file_content(dir_path / "file2")

    _verify_with_dir_modification(dir_hash_test_dir, mod_file1, True, ["file1"])
    _verify_with_dir_modification(dir_hash_test_dir, mod_file2, False, ["file1"])


def test_dir_hash_ignore_pattern_multiple(dir_hash_test_dir):
    _update_file_content(dir_hash_test_dir / "file1")
    _update_file_content(dir_hash_test_dir / "file2")

    def mod(dir_path):
        _update_file_content(dir_path / "file1")
        _update_file_content(dir_path / "file2")

    _verify_with_dir_modification(dir_hash_test_dir, mod, True, ["file1", "file2"])


def test_dir_hash_ignore_pattern_dir_glob(dir_hash_test_dir):
    tmp_dir = dir_hash_test_dir / "tmp_dir"
    tmp_dir.mkdir()
    _update_file_content(tmp_dir / "file1")
    _update_file_content(tmp_dir / "file2")

    def mod(dir_path):
        _update_file_content(dir_path / "tmp_dir" / "file1")
        _update_file_content(dir_path / "tmp_dir" / "file2")

    _verify_with_dir_modification(dir_hash_test_dir, mod, True, ["tmp_dir/*"])


def test_file_content_hash(tmp_path):
    orig_content = _generate_random_string(1024 * 1024)
    file_path = tmp_path / "file"
    _update_file_content(file_path, orig_content)
    orig_hash = file_content_hash(file_path)
    new_content = _generate_random_string(2048 * 1024)
    _update_file_content(file_path, new_content)
    new_hash = file_content_hash(file_path)
    assert orig_hash != new_hash

    _update_file_content(file_path, orig_content)
    final_hash = file_content_hash(file_path)
    assert final_hash == orig_hash


def test_file_content_hash_str(tmp_path):
    orig_content = _generate_random_string(1024 * 1024)
    file_path = tmp_path / "file"
    _update_file_content(file_path, orig_content)
    orig_hash = file_content_hash_str(file_path)
    new_content = _generate_random_string(2048 * 1024)
    _update_file_content(file_path, new_content)
    new_hash = file_content_hash_str(file_path)
    assert orig_hash != new_hash

    _update_file_content(file_path, orig_content)
    final_hash = file_content_hash_str(file_path)
    assert final_hash == orig_hash


def _verify_with_dir_modification(
    target_dir: Path,
    op: Callable[[Path], Path],
    should_match: bool,
    ignore_patterns: List[str] = None,
):
    hash1 = directory_content_hash(target_dir, ignore_patterns=ignore_patterns)
    new_target_dir = op(target_dir)
    hash2 = directory_content_hash(
        new_target_dir or target_dir, ignore_patterns=ignore_patterns
    )
    if should_match:
        assert hash1 == hash2
    else:
        assert hash1 != hash2


def _generate_random_string(size: int = 10) -> str:
    letters = string.ascii_letters
    return "".join(random.choice(letters) for _ in range(size))


def _update_file_content(file: Path, content: str = None):
    if content is None:
        content = _generate_random_string()
    with file.open("w") as f:
        f.write(content)

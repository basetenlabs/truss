from truss.patch.dir_signature import directory_content_signature


def test_directory_content_signature(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "file1").touch()
    (root / "file2").touch()
    subdir = root / "dir"
    subdir.mkdir()
    (subdir / "file3").touch()

    content_sign = directory_content_signature(root)

    assert content_sign.keys() == {
        "dir",
        "dir/file3",
        "file1",
        "file2",
    }


def test_directory_content_signature_no_data_dir(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "file1").touch()
    (root / "file2").touch()
    data_dir = root / "data"
    data_dir.mkdir()
    (data_dir / "file3").touch()

    content_sign = directory_content_signature(root=root, ignore_patterns=["data/*"])

    assert content_sign.keys() == {
        "data",
        "file1",
        "file2",
    }

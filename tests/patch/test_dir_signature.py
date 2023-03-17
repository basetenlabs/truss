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
    print(content_sign)

    assert content_sign.keys() == {
        "dir",
        "dir/file3",
        "file1",
        "file2",
    }

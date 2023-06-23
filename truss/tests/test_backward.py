from truss.build import from_directory, mk_truss


def test_mk_truss_passthrough(sklearn_rfc_model, tmp_path):
    dir_path = tmp_path / "truss"
    data_file_path = tmp_path / "data.txt"
    with data_file_path.open("w") as data_file:
        data_file.write("test")
    req_file_path = tmp_path / "requirements.txt"
    requirements = [
        "tensorflow==2.3.1",
        "uvicorn==0.12.2",
    ]
    with req_file_path.open("w") as req_file:
        for req in requirements:
            req_file.write(f"{req}\n")
    scaf = mk_truss(
        sklearn_rfc_model,
        target_directory=dir_path,
        data_files=[str(data_file_path)],
        requirements_file=str(req_file_path),
    )
    spec = scaf.spec
    assert spec.model_module_dir.exists()
    assert spec.truss_dir == dir_path
    assert spec.config_path.exists()
    assert spec.data_dir.exists()
    assert (spec.data_dir / "data.txt").exists()
    assert spec.requirements == requirements


def test_from_directory_passthrough(sklearn_rfc_model, tmp_path):
    dir_path = tmp_path / "truss"
    data_file_path = tmp_path / "data.txt"
    with data_file_path.open("w") as data_file:
        data_file.write("test")
    req_file_path = tmp_path / "requirements.txt"
    requirements = [
        "tensorflow==2.3.1",
        "uvicorn==0.12.2",
    ]
    with req_file_path.open("w") as req_file:
        for req in requirements:
            req_file.write(f"{req}\n")
    mk_truss(
        sklearn_rfc_model,
        target_directory=dir_path,
        data_files=[str(data_file_path)],
        requirements_file=str(req_file_path),
    )
    scaf = from_directory(dir_path)
    spec = scaf.spec
    assert spec.model_module_dir.exists()
    assert spec.truss_dir == dir_path
    assert spec.config_path.exists()
    assert spec.data_dir.exists()
    assert (spec.data_dir / "data.txt").exists()
    assert spec.requirements == requirements

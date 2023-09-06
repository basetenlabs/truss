from pathlib import Path

from truss.build import init
from truss.truss_spec import TrussSpec


def test_truss_init(tmp_path):
    dir_name = str(tmp_path)
    init(dir_name)
    spec = TrussSpec(Path(dir_name))
    assert spec.model_module_dir.exists()
    assert spec.data_dir.exists()
    assert spec.truss_dir == tmp_path
    assert spec.config_path.exists()


def test_truss_init_with_data_file_and_requirements_file_and_bundled_packages(
    tmp_path,
):
    dir_path = tmp_path / "truss"
    dir_name = str(dir_path)

    # Init data files
    data_path = tmp_path / "data.txt"
    with data_path.open("w") as data_file:
        data_file.write("test")

    # Init requirements file
    req_file_path = tmp_path / "requirements.txt"
    requirements = [
        "tensorflow==2.3.1",
        "uvicorn==0.12.2",
    ]
    with req_file_path.open("w") as req_file:
        for req in requirements:
            req_file.write(f"{req}\n")

    # init bundled packages
    packages_path = tmp_path / "dep_pkg"
    packages_path.mkdir()
    packages_path_file_py = packages_path / "file.py"
    packages_path_init_py = packages_path / "__init__.py"
    pkg_files = [packages_path_init_py, packages_path_file_py]
    for pkg_file in pkg_files:
        with pkg_file.open("w") as fh:
            fh.write("test")

    init(
        dir_name,
        data_files=[str(data_path)],
        requirements_file=str(req_file_path),
        bundled_packages=[str(packages_path)],
    )
    spec = TrussSpec(Path(dir_name))
    assert spec.model_module_dir.exists()
    assert spec.truss_dir == dir_path
    assert spec.config_path.exists()
    assert spec.data_dir.exists()
    assert spec.bundled_packages_dir.exists()
    assert (spec.data_dir / "data.txt").exists()
    assert spec.requirements == requirements
    assert (spec.bundled_packages_dir / "dep_pkg" / "__init__.py").exists()
    assert (spec.bundled_packages_dir / "dep_pkg" / "file.py").exists()

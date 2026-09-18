import logging
import pathlib

import pytest

from truss.base.truss_config import RequirementsFileType
from truss_chains import public_types
from truss_chains.deployment import code_gen


@pytest.fixture
def tmp_chainlet_dir(tmp_path):
    """Create a temporary chainlet directory."""
    chainlet_dir = tmp_path / "chainlet"
    chainlet_dir.mkdir()
    return chainlet_dir


@pytest.fixture
def tmp_requirements_txt(tmp_path):
    """Create a temporary requirements.txt file."""
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("numpy>=1.21\nrequests\n")
    return req_file


@pytest.fixture
def tmp_pyproject_toml(tmp_path):
    """Create a temporary pyproject.toml file."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "my-chain"\nversion = "0.1.0"\n'
        'dependencies = [\n  "numpy>=1.21",\n  "requests",\n'
        f'  "truss=={__import__("truss").__version__}",\n]\n'
    )
    return pyproject


@pytest.fixture
def tmp_uv_lock(tmp_path):
    """Create a temporary uv.lock and its sibling pyproject.toml."""
    uv_lock = tmp_path / "uv.lock"
    uv_lock.write_text("version = 1\n")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "my-chain"\nversion = "0.1.0"\n'
        'dependencies = [\n  "numpy>=1.21",\n  "requests",\n'
        f'  "truss=={__import__("truss").__version__}",\n]\n'
    )
    return uv_lock


def _make_abs_path(path: pathlib.Path) -> public_types.AbsPath:
    return public_types.AbsPath(
        abs_file_path=str(path), creating_module=__name__, original_path=str(path)
    )


def test_pip_requirements_file_emits_deprecation_warning(tmp_requirements_txt, caplog):
    with caplog.at_level(logging.WARNING):
        image = public_types.DockerImage(
            pip_requirements_file=_make_abs_path(tmp_requirements_txt)
        )
    assert "deprecated" in caplog.text
    assert image.requirements_file is not None
    assert image.requirements_file.abs_path == str(tmp_requirements_txt)


def test_both_pip_requirements_file_and_requirements_file_raises(
    tmp_requirements_txt, tmp_pyproject_toml
):
    with pytest.raises(public_types.ChainsUsageError, match="Cannot specify both"):
        public_types.DockerImage(
            pip_requirements_file=_make_abs_path(tmp_requirements_txt),
            requirements_file=_make_abs_path(tmp_pyproject_toml),
        )


def test_requirements_file_no_deprecation_warning(tmp_requirements_txt, caplog):
    with caplog.at_level(logging.WARNING):
        image = public_types.DockerImage(
            requirements_file=_make_abs_path(tmp_requirements_txt)
        )
    assert "deprecated" not in caplog.text
    assert image.requirements_file is not None


def test_detect_pip_format(tmp_requirements_txt):
    image = public_types.DockerImage(
        requirements_file=_make_abs_path(tmp_requirements_txt)
    )
    assert code_gen._detect_requirements_file_type(image) == RequirementsFileType.PIP


def test_detect_pyproject_format(tmp_pyproject_toml):
    image = public_types.DockerImage(
        requirements_file=_make_abs_path(tmp_pyproject_toml)
    )
    assert (
        code_gen._detect_requirements_file_type(image) == RequirementsFileType.PYPROJECT
    )


def test_detect_uv_lock_format(tmp_uv_lock):
    image = public_types.DockerImage(requirements_file=_make_abs_path(tmp_uv_lock))
    assert (
        code_gen._detect_requirements_file_type(image) == RequirementsFileType.UV_LOCK
    )


def test_detect_not_provided():
    image = public_types.DockerImage()
    assert (
        code_gen._detect_requirements_file_type(image)
        == RequirementsFileType.NOT_PROVIDED
    )


def test_make_pip_requirements_merges_file_and_list(tmp_requirements_txt):
    image = public_types.DockerImage(
        requirements_file=_make_abs_path(tmp_requirements_txt),
        pip_requirements=["scipy"],
    )
    reqs = code_gen._make_pip_requirements(image)
    assert "numpy>=1.21" in reqs
    assert "requests" in reqs
    assert "scipy" in reqs
    assert any(r.startswith("truss==") for r in reqs)


def test_make_pip_requirements_list_only():
    image = public_types.DockerImage(pip_requirements=["numpy"])
    reqs = code_gen._make_pip_requirements(image)
    assert "numpy" in reqs
    assert any(r.startswith("truss==") for r in reqs)


def test_pyproject_with_pip_requirements_raises(tmp_pyproject_toml, tmp_chainlet_dir):
    image = public_types.DockerImage(
        requirements_file=_make_abs_path(tmp_pyproject_toml),
        pip_requirements=["extra-package"],
    )
    with pytest.raises(
        public_types.ChainsUsageError, match="pip_requirements.*cannot be used"
    ):
        code_gen._prepare_pyproject_requirements(
            image, tmp_chainlet_dir, RequirementsFileType.PYPROJECT
        )


def test_pyproject_auto_adds_truss(tmp_path, tmp_chainlet_dir):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "my-chain"\nversion = "0.1.0"\n'
        'dependencies = [\n  "numpy>=1.21",\n]\n'
    )
    image = public_types.DockerImage(requirements_file=_make_abs_path(pyproject))
    code_gen._prepare_pyproject_requirements(
        image, tmp_chainlet_dir, RequirementsFileType.PYPROJECT
    )
    import tomlkit

    copied = tmp_chainlet_dir / "pyproject.toml"
    with open(copied) as f:
        doc = tomlkit.load(f)
    deps = doc["project"]["dependencies"]
    assert any("truss==" in dep for dep in deps)
    # Original file should be unchanged.
    with open(pyproject) as f:
        original = tomlkit.load(f)
    assert not any("truss==" in dep for dep in original["project"]["dependencies"])


def test_pyproject_does_not_add_truss_if_present(tmp_pyproject_toml, tmp_chainlet_dir):
    image = public_types.DockerImage(
        requirements_file=_make_abs_path(tmp_pyproject_toml)
    )
    code_gen._prepare_pyproject_requirements(
        image, tmp_chainlet_dir, RequirementsFileType.PYPROJECT
    )
    import tomlkit

    copied = tmp_chainlet_dir / "pyproject.toml"
    with open(copied) as f:
        doc = tomlkit.load(f)
    deps = doc["project"]["dependencies"]
    truss_deps = [dep for dep in deps if "truss" in dep]
    assert len(truss_deps) == 1

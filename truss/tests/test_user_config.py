import subprocess
import sys


def test_import_truss_does_not_create_config_dir(tmp_path):
    """Regression test: `import truss` must not create ~/.config/truss as a side effect."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, os; "
                "import truss; "
                "config_dir = pathlib.Path.home() / '.config' / 'truss'; "
                "assert not config_dir.exists(), "
                "f'{config_dir} was created as an import side effect'"
            ),
        ],
        env={**dict(HOME=str(tmp_path)), "PATH": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

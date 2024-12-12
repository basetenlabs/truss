# type: ignore  # This tool is only for Marius.
"""Script to auot-generate the API reference for Truss Chains."""

import inspect
import pathlib
import shutil
import subprocess
import tempfile
from pathlib import Path

from sphinx import application

import truss_chains as chains

DUMMY_INDEX_RST = """
.. Dummy

Welcome to Truss Chains's documentation!
========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
"""


BUILDER = "mdx_adapter"  # "html"
NON_PUBLIC_SYMBOLS = [
    # "truss_chains.definitions.AssetSpec",
    # "truss_chains.definitions.ComputeSpec",
    "truss_chains.deployment.deployment_client.ChainService",
    "truss_chains.definitions.Environment",
]


SECTION_CHAINLET = (
    "Chainlet classes",
    "APIs for creating user-defined Chainlets.",
    [
        "truss_chains.ChainletBase",
        "truss_chains.depends",
        "truss_chains.depends_context",
        "truss_chains.DeploymentContext",
        "truss_chains.definitions.Environment",
        "truss_chains.ChainletOptions",
        "truss_chains.RPCOptions",
        "truss_chains.mark_entrypoint",
    ],
)
SECTION_CONFIG = (
    "Remote Configuration",
    (
        "These data structures specify for each chainlet how it gets deployed "
        "remotely, e.g. dependencies and compute resources."
    ),
    [
        "truss_chains.RemoteConfig",
        "truss_chains.DockerImage",
        "truss_chains.BasetenImage",
        "truss_chains.CustomImage",
        "truss_chains.Compute",
        "truss_chains.Assets",
    ],
)
SECTION_UTILITIES = (
    "Core",
    "General framework and helper functions.",
    [
        "truss_chains.push",
        "truss_chains.deployment.deployment_client.ChainService",
        "truss_chains.make_abs_path_here",
        "truss_chains.run_local",
        "truss_chains.DeployedServiceDescriptor",
        "truss_chains.StubBase",
        "truss_chains.RemoteErrorDetail",
        # "truss_chains.ChainsRuntimeError",
    ],
)

SECTIONS = [SECTION_CHAINLET, SECTION_CONFIG, SECTION_UTILITIES]


def _list_imported_symbols(module: object) -> dict[str, str]:
    imported_symbols = {
        f"truss_chains.{name}": (
            "autoclass"
            if inspect.isclass(obj)
            else "autofunction"
            if inspect.isfunction(obj)
            else "autodata"
        )
        for name, obj in inspect.getmembers(module)
        if not name.startswith("_") and not inspect.ismodule(obj)
    }
    # Extra classes that are not really exported as public API, but are still relevant.
    imported_symbols.update({sym: "autoclass" for sym in NON_PUBLIC_SYMBOLS})
    return imported_symbols


def _make_rst_structure(chains):
    exported_symbols = _list_imported_symbols(chains)
    rst_parts = []
    for name, descr, symbols in SECTIONS:
        rst_parts.append(name)
        rst_parts.append("=" * len(rst_parts[-1]) + "\n")
        rst_parts.append(descr)
        rst_parts.append("\n")

        for symbol in symbols:
            kind = exported_symbols.pop(symbol)
            rst_parts.append(f".. {kind}:: {symbol}")
            rst_parts.append("\n")

    if exported_symbols:
        raise ValueError(
            "All symbols must be mapped to a section. Left over:"
            f"{list(exported_symbols.keys())}."
        )
    return "\n".join(rst_parts)


def _clean_build_directory(build_dir: Path) -> None:
    if build_dir.exists() and build_dir.is_dir():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)


def _apply_patch(
    original_file_path: str, patch_file_path: str, output_file_path: str
) -> None:
    original_file = Path(original_file_path)
    patch_file = Path(patch_file_path)
    output_file = Path(output_file_path)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_output_file_path = Path(temp_file.name)

    try:
        subprocess.run(
            [
                "patch",
                str(original_file),
                "-o",
                str(temp_output_file_path),
                str(patch_file),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Copy temp file to final output if no errors
        shutil.copy(temp_output_file_path, output_file)

    except subprocess.CalledProcessError as e:
        reject_file = temp_output_file_path.with_suffix(".rej")
        if reject_file.exists():
            print(f"Conflicts found, saved to {reject_file}")
            subprocess.run(
                [
                    "meld",
                    str(original_file_path),
                    str(output_file),
                    str(temp_output_file_path),
                ],
                check=True,
            )
        else:
            print(f"Patch failed: {e.stderr}")

    finally:
        if temp_output_file_path.exists():
            temp_output_file_path.unlink()


def generate_sphinx_docs(output_dir: pathlib.Path) -> None:
    _clean_build_directory(output_dir)
    config_file = pathlib.Path(__file__).parent / "sphinx_config.py"
    docs_dir = output_dir / "docs"
    conf_dir = docs_dir
    doctree_dir = docs_dir / "doctrees"

    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "conf.py").write_text(config_file.read_text())
    (docs_dir / "index.rst").write_text(DUMMY_INDEX_RST)
    (docs_dir / "modules.rst").write_text(_make_rst_structure(chains))

    app = application.Sphinx(
        srcdir=str(docs_dir),
        confdir=str(conf_dir),
        outdir=str(Path(output_dir).resolve()),
        doctreedir=str(doctree_dir),
        buildername=BUILDER,
    )
    app.build()
    if BUILDER == "mdx_adapter":
        dog_gen_dir = pathlib.Path(__file__).parent.absolute()
        generated_reference_path = dog_gen_dir / "generated-reference.mdx"
        shutil.copy(output_dir / "modules.mdx", generated_reference_path)
        patch_file_path = dog_gen_dir / "reference.patch"
        # Apply patch to generated_reference_path
        patched_reference_path = dog_gen_dir / "API-reference.mdx"
        _apply_patch(
            str(generated_reference_path),
            str(patch_file_path),
            str(patched_reference_path),
        )


if __name__ == "__main__":
    generate_sphinx_docs(
        output_dir=pathlib.Path("/tmp/doc_gen"),
    )

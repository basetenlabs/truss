from typing import List

import pandas as pd  # noqa
from truss.build import create


def test_infer_deps_through_create_local_import(sklearn_rfc_model, tmp_path):
    dir_path = tmp_path / "truss"
    import requests  # noqa

    tr = create(
        sklearn_rfc_model,
        target_directory=dir_path,
    )
    spec = tr.spec
    _validate_that_package_is_in_requirements(spec.requirements, "requests")


def test_infer_deps_through_create_top_of_the_file_import(sklearn_rfc_model, tmp_path):
    dir_path = tmp_path / "truss"
    tr = create(
        sklearn_rfc_model,
        target_directory=dir_path,
    )
    spec = tr.spec
    _validate_that_package_is_in_requirements(spec.requirements, "pandas")


def _validate_that_package_is_in_requirements(
    requirements_list: List[str],
    package_name: str,
):
    requirement_entries = [
        req for req in requirements_list if req.startswith(package_name)
    ]
    assert len(requirement_entries) == 1

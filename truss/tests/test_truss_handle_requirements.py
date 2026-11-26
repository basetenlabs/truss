"""Requirement updates must preserve pip's supported Git URL inputs (#231, #252)."""

import pytest

from truss.truss_handle.truss_handle import TrussHandle


@pytest.mark.parametrize(
    "existing, added, expected",
    [
        ([], "requests", ["requests"]),
        (["requests==2.31.0"], "requests==2.31.0", ["requests==2.31.0"]),
        (["Requests<3", "numpy"], "requests>=2.32", ["numpy", "requests>=2.32"]),
        (["my_pkg==1", "other"], "my-pkg==2", ["other", "my-pkg==2"]),
        (["requests==1", "Requests==2"], "requests==3", ["requests==3"]),
        (
            [
                'requests==1; python_version < "3.10"',
                'requests==2; python_version >= "3.10"',
            ],
            'requests==3; python_version >= "3.10"',
            [
                'requests==1; python_version < "3.10"',
                'requests==3; python_version >= "3.10"',
            ],
        ),
        (
            ["requests[socks]==1", "requests[security]==2"],
            "requests[socks]==3",
            ["requests[security]==2", "requests[socks]==3"],
        ),
        (
            ["model_utils[Extra_A]==1"],
            "model-utils[extra-a]==2",
            ["model-utils[extra-a]==2"],
        ),
        (
            ["model-utils==1"],
            "model-utils @ git+https://github.com/example/model-utils.git@v2",
            ["model-utils @ git+https://github.com/example/model-utils.git@v2"],
        ),
        (
            ["model-utils @ https://example.test/model-utils-1.tar.gz"],
            "model-utils==2",
            ["model-utils==2"],
        ),
    ],
)
def test_requirement_updates_persist_without_changing_other_dependency_scopes(
    custom_model_truss_dir_with_pre_and_post, existing, added, expected
):
    path = custom_model_truss_dir_with_pre_and_post
    handle = TrussHandle(path)
    handle.update_requirements(existing)

    handle.add_python_requirement(added)

    assert TrussHandle(path).spec.requirements == expected


@pytest.mark.parametrize(
    "opaque",
    [
        "git+https://github.com/example/model-utils.git@main#egg=model-utils",
        "git+ssh://git@github.com/example/model-utils.git",
        "https://example.test/model-utils-1.tar.gz",
        "./local-package",
        "-e ../editable-package",
    ],
)
def test_opaque_pip_requirement_is_preserved_and_exact_addition_is_idempotent(
    custom_model_truss_dir_with_pre_and_post, opaque
):
    path = custom_model_truss_dir_with_pre_and_post
    handle = TrussHandle(path)
    handle.update_requirements(["requests", opaque])

    handle.add_python_requirement(opaque)
    handle.add_python_requirement("numpy")

    assert TrussHandle(path).spec.requirements == ["requests", opaque, "numpy"]


def test_add_requirement_replaces_version_without_rejecting_git_requirements(
    custom_model_truss_dir_with_pre_and_post,
):
    path = custom_model_truss_dir_with_pre_and_post
    handle = TrussHandle(path)
    git_requirement = (
        "git+https://github.com/example/model-utils.git@main#egg=model-utils"
    )
    handle.update_requirements([git_requirement, "requests==2.31.0", "numpy==1.26.0"])

    handle.add_python_requirement("requests==2.32.0")

    assert TrussHandle(path).spec.requirements == [
        git_requirement,
        "numpy==1.26.0",
        "requests==2.32.0",
    ]

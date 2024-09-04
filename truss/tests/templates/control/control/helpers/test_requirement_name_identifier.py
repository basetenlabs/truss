import pytest
from truss.templates.control.control.helpers.truss_patch.requirement_name_identifier import (
    RequirementMeta,
    identify_requirement_name,
    reqs_by_name,
)


@pytest.mark.parametrize(
    "req, expected_name",
    [
        ("pytorch", "pytorch"),
        (
            "git+https://github.com/huggingface/transformers.git#egg=transformers",
            "git+github.com/huggingface/transformers.git",
        ),
        (
            "git+https://github.com/huggingface/transformers.git",
            "git+github.com/huggingface/transformers.git",
        ),
        (
            "git+https://github.com/huggingface/transformers.git@main#egg=transformers",
            "git+github.com/huggingface/transformers.git",
        ),
        (
            " git+https://github.com/huggingface/transformers.git ",
            "git+github.com/huggingface/transformers.git",
        ),
        ("pytorch==1.0", "pytorch"),
        ("pytorch>=1.0", "pytorch"),
        ("pytorch<=1.0", "pytorch"),
    ],
)
def test_identify_requirement_name(req, expected_name):
    assert expected_name == identify_requirement_name(req)


def test_reqs_by_name():
    reqs = [
        "pytorch",
        " ",
        "jinja==1.0",
    ]
    assert reqs_by_name(reqs) == {"pytorch": "pytorch", "jinja": "jinja==1.0"}


@pytest.mark.parametrize(
    "desc, req, expected_meta",
    [
        (
            "handles simple requirement",
            "pytorch",
            RequirementMeta(
                requirement="pytorch",
                name="pytorch",
                is_url_based_requirement=False,
                egg_tag=None,
            ),
        ),
        (
            "handles python package with version",
            "pytorch==1.0",
            RequirementMeta(
                requirement="pytorch==1.0",
                name="pytorch",
                is_url_based_requirement=False,
                egg_tag=None,
            ),
        ),
        (
            "handles url-based requirement with egg tag",
            "git+https://github.com/huggingface/transformers.git@main#egg=transformers",
            RequirementMeta(
                requirement="git+https://github.com/huggingface/transformers.git@main#egg=transformers",
                name="git+github.com/huggingface/transformers.git",
                is_url_based_requirement=True,
                egg_tag=["transformers"],
            ),
        ),
        (
            "handles url-based requirement without egg tag",
            "git+https://github.com/huggingface/transformers.git",
            RequirementMeta(
                requirement="git+https://github.com/huggingface/transformers.git",
                name="git+github.com/huggingface/transformers.git",
                is_url_based_requirement=True,
                egg_tag=None,
            ),
        ),
    ],
)
def test_requirement_meta_from_req(desc, req: str, expected_meta: RequirementMeta):
    assert expected_meta == RequirementMeta.from_req(req), desc

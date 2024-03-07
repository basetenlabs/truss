import pytest
from truss.templates.control.control.helpers.truss_patch.requirement_name_identifier import (
    identify_requirement_name,
    reqs_by_name,
)


@pytest.mark.parametrize(
    "req, expected_name",
    [
        ("pytorch", "pytorch"),
        (
            "git+https://github.com/huggingface/transformers.git",
            "git+https://github.com/huggingface/transformers.git",
        ),
        (
            " git+https://github.com/huggingface/transformers.git ",
            "git+https://github.com/huggingface/transformers.git",
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
        "jinja==1.0",
    ]
    assert reqs_by_name(reqs) == {"pytorch": "pytorch", "jinja": "jinja==1.0"}

import pytest

from truss.base.errors import ValidationError
from truss.base.truss_spec import TrussSpec
from truss.trt_llm.validation import _verify_has_class_init_arg, validate


@pytest.mark.parametrize(
    "src, class_name, expected_arg, expected_to_raise",
    [
        (
            """
class Model:
    def __init__(self, foo):
        pass
            """,
            "Model",
            "foo",
            False,
        ),
        # Missing arg
        (
            """
class Model:
    def __init__(self, foo):
        pass
            """,
            "Model",
            "bar",
            True,
        ),
        # Missing class
        (
            """
class Model:
    def __init__(self, foo):
        pass
            """,
            "Missing",
            "bar",
            True,
        ),
        # With default value should still work
        (
            """
class Model:
    def __init__(self, foo=None):
        pass
            """,
            "Model",
            "foo",
            False,
        ),
        # not the only arg
        (
            """
class Model:
    def __init__(self, first, foo, third):
        pass
            """,
            "Model",
            "foo",
            False,
        ),
        # Class with class-level annotated assignments before __init__ (mirrors
        # the chains-generated ``TrussChainletModel``). Should not crash on
        # accessing ``.name`` of an ``ast.AnnAssign`` node.
        (
            """
from typing import Any, Optional


class TrussChainletModel:
    _context: Any
    _chainlet: Any
    _trt_llm: Optional[Any]

    def __init__(self, config, trt_llm=None):
        pass
            """,
            "TrussChainletModel",
            "trt_llm",
            False,
        ),
        # Same shape, but missing the ``trt_llm`` arg -- should raise the
        # normal "missing arg" ValidationError, not an AttributeError.
        (
            """
class TrussChainletModel:
    _context: object
    _chainlet: object

    def __init__(self, config):
        pass
            """,
            "TrussChainletModel",
            "trt_llm",
            True,
        ),
    ],
)
def test_has_class_init_arg(src, expected_arg, class_name, expected_to_raise):
    if expected_to_raise:
        with pytest.raises(ValidationError):
            _verify_has_class_init_arg(src, class_name, expected_arg)
    else:
        _verify_has_class_init_arg(src, class_name, expected_arg)


def test_validate(custom_model_trt_llm):
    spec = TrussSpec(custom_model_trt_llm)
    validate(spec)

    # overwrite with model code that doesn't take trt_llm as input
    override_model_code_invalid = """
class Model:
    def __init__(self):
        pass
"""
    spec.model_class_filepath.write_text(override_model_code_invalid)
    new_spec = TrussSpec(custom_model_trt_llm)
    with pytest.raises(ValidationError):
        validate(new_spec)

    # If model class file is removed, it should be ok
    spec.model_class_filepath.unlink()
    validate(new_spec)

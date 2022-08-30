import pytest
from truss.constants import PYTORCH
from truss.model_inference import (
    infer_model_information,
    validate_provided_parameters_with_model,
)


def test_pytorch_init_arg_validation(
    pytorch_model_with_init_args, pytorch_model_init_args
):
    pytorch_model_with_init_args, _ = pytorch_model_with_init_args
    # Validates with args and kwargs
    validate_provided_parameters_with_model(
        pytorch_model_with_init_args.__class__, pytorch_model_init_args
    )

    # Errors if bad args
    with pytest.raises(ValueError):
        validate_provided_parameters_with_model(
            pytorch_model_with_init_args.__class__, {"foo": "bar"}
        )

    # Validates with only args
    copied_args = pytorch_model_init_args.copy()
    copied_args.pop("kwarg1")
    copied_args.pop("kwarg2")
    validate_provided_parameters_with_model(pytorch_model_with_init_args, copied_args)

    # Requires all args
    with pytest.raises(ValueError):
        validate_provided_parameters_with_model(pytorch_model_with_init_args, {})


def test_infer_model_information(pytorch_model_with_init_args):
    model_info = infer_model_information(pytorch_model_with_init_args[0])
    assert model_info.model_framework == PYTORCH
    assert model_info.model_type == "MyModel"

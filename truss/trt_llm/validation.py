import ast

from truss.errors import ValidationError
from truss.truss_spec import TrussSpec


def validate(truss_spec: TrussSpec):
    model_class_filepath = truss_spec.model_class_filepath
    if not model_class_filepath.exists():
        # It's ok if model class file is not provided,
        # trt_llm will generate one.
        return

    source = model_class_filepath.read_text()
    _verify_has_class_init_arg(source, truss_spec.model_class_name, "trt_llm")


def _verify_has_class_init_arg(source: str, class_name: str, arg_name: str):
    tree = ast.parse(source)
    model_class_init_found = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if child.name == "__init__":  # type: ignore[attr-defined]
                    model_class_init_found = True
                    arg_found = False
                    for arg in child.args.args:  # type: ignore[attr-defined]
                        if arg.arg == arg_name:
                            arg_found = True
                    if not arg_found:
                        raise ValidationError(
                            (
                                "Model class `__init__` method is required to have `trt_llm` as an argument. Please add that argument.\n  "
                                "Or if you want to use the automatically generated model class then remove the `model.py` file."
                            )
                        )

    if not model_class_init_found:
        raise ValidationError(
            "Model class does not have an `__init__` method; when using `trt_llm` it is required"
        )

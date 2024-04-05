import logging
import pathlib
import textwrap
from typing import Any

import libcst
from slay import definitions, utils
from slay.truss_adapter import model_skeleton

INDENT = " " * 4


def _indent(text: str, num: int = 1) -> str:
    return textwrap.indent(text, INDENT * num)


class _SpecifyProcessorTypeAnnotation(libcst.CSTTransformer):
    def __init__(self, new_annotation: str) -> None:
        super().__init__()
        self._new_annotation = new_annotation

    def leave_SimpleStatementLine(
        self,
        original_node: libcst.SimpleStatementLine,
        updated_node: libcst.SimpleStatementLine,
    ) -> libcst.SimpleStatementLine:
        new_body: list[Any] = []
        for statement in updated_node.body:
            if (
                isinstance(statement, libcst.AnnAssign)
                and isinstance(statement.target, libcst.Name)
                and statement.target.value == "_processor"
            ):
                new_annotation = libcst.Annotation(
                    annotation=libcst.Name(value=self._new_annotation)
                )
                new_statement = statement.with_changes(annotation=new_annotation)
                new_body.append(new_statement)
            else:
                new_body.append(statement)

        return updated_node.with_changes(body=tuple(new_body))


def _gen_load_src(processor_name: str):
    """Generates AST for the `load` method of the truss model."""
    body = _indent(
        "\n".join(
            [
                f'logging.info(f"Loading processor `{processor_name}`.")',
                f"self._processor = {processor_name}(context=self._context)",
            ]
        )
    )
    return libcst.parse_statement("\n".join(["def load(self) -> None:", body]))


def _gen_predict_src(
    endpoint_descriptor: definitions.EndpointAPIDescriptor, processor_name: str
):
    """Generates AST for the `predict` method of the truss model."""
    if endpoint_descriptor.is_generator:
        # TODO: implement generator.
        raise NotImplementedError("Generator.")

    parts = []
    def_str = "async def" if endpoint_descriptor.is_async else "def"
    parts.append(f"{def_str} predict(self, payload):")
    # Add error handling context manager:
    parts.append(
        _indent(
            f"with utils.exception_to_http_error("
            f'include_stack=True, processor_name="{processor_name}"):'
        )
    )
    # Convert items from json payload dict to an arg-list, parsing pydantic models.
    args = ", ".join(
        (
            f"{arg_name}={arg_type.as_src_str()}.parse_obj(payload['{arg_name}'])"
            if arg_type.is_pydantic
            else f"{arg_name}=payload['{arg_name}']"
        )
        for arg_name, arg_type in endpoint_descriptor.input_names_and_types
    )
    # Invoke processor.
    maybe_await = "await " if endpoint_descriptor.is_async else ""
    parts.append(
        _indent(
            f"result = {maybe_await}self._processor.{endpoint_descriptor.name}({args})",
            2,
        )
    )
    # Return as json tuple, serialize pydantic models.
    if len(endpoint_descriptor.output_types) == 1:
        output_type = endpoint_descriptor.output_types[0]
        result = "result.dict()" if output_type.is_pydantic else "result"
    else:
        result_parts = [
            f"result[{i}].dict()" if t.is_pydantic else f"result[{i}]"
            for i, t in enumerate(endpoint_descriptor.output_types)
        ]
        result = f"{', '.join(result_parts)}"

    parts.append(_indent(f"return {result}"))

    return libcst.parse_statement("\n".join(parts))


def generate_truss_model(
    processor_descriptor: definitions.ProcessorAPIDescriptor,
) -> tuple[libcst.CSTNode, list[libcst.SimpleStatementLine], libcst.CSTNode]:
    logging.info(f"Generating Truss model for `{processor_descriptor.cls_name}`.")
    skeleton_tree = libcst.parse_module(
        pathlib.Path(model_skeleton.__file__).read_text()
    )
    imports: list[Any] = [
        node
        for node in skeleton_tree.body
        if isinstance(node, libcst.SimpleStatementLine)
        and any(
            isinstance(stmt, libcst.Import) or isinstance(stmt, libcst.ImportFrom)
            for stmt in node.body
        )
    ]
    imports.append(libcst.parse_statement("import logging"))
    imports.append(libcst.parse_statement("from slay import utils"))

    class_definition: libcst.ClassDef = utils.expect_one(
        node
        for node in skeleton_tree.body
        if isinstance(node, libcst.ClassDef)
        and node.name.value == model_skeleton.ProcessorModel.__name__
    )

    load_def = _gen_load_src(processor_descriptor.cls_name)
    predict_def = _gen_predict_src(
        processor_descriptor.endpoint, processor_descriptor.cls_name
    )
    new_body: list[Any] = list(class_definition.body.body) + [load_def, predict_def]
    new_block = libcst.IndentedBlock(body=new_body)
    class_definition = class_definition.with_changes(body=new_block)
    class_definition = class_definition.visit(  # type: ignore[assignment]
        _SpecifyProcessorTypeAnnotation(processor_descriptor.cls_name)
    )
    if issubclass(processor_descriptor.user_config_type.raw, type(None)):
        userconfig_pin = libcst.parse_statement("UserConfigT = None")
    else:
        userconfig_pin = libcst.parse_statement(
            f"UserConfigT = {processor_descriptor.user_config_type.as_src_str()}"
        )
    return class_definition, imports, userconfig_pin

import logging
import pathlib
from typing import Any

import libcst
from slay import definitions, utils
from slay.truss_adapter import model_skeleton


class _SpecifyProcessorTypeAnnotation(libcst.CSTTransformer):
    def __init__(self, new_annotaiton: str) -> None:
        super().__init__()
        self._new_annotaiton = new_annotaiton

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
                    annotation=libcst.Name(value=self._new_annotaiton)
                )
                new_statement = statement.with_changes(annotation=new_annotation)
                new_body.append(new_statement)
            else:
                new_body.append(statement)

        return updated_node.with_changes(body=tuple(new_body))


def generate_truss_model(
    processor_desrciptor: definitions.ProcessorAPIDescriptor,
) -> tuple[libcst.CSTNode, list[libcst.SimpleStatementLine], libcst.CSTNode]:
    logging.info(f"Generating Baseten model for `{processor_desrciptor.cls_name}`.")
    skeleton_tree = libcst.parse_module(
        pathlib.Path(model_skeleton.__file__).read_text()
    )

    imports = [
        node
        for node in skeleton_tree.body
        if isinstance(node, libcst.SimpleStatementLine)
        and any(
            isinstance(stmt, libcst.Import) or isinstance(stmt, libcst.ImportFrom)
            for stmt in node.body
        )
    ]

    class_definition: libcst.ClassDef = utils.expect_one(
        node
        for node in skeleton_tree.body
        if isinstance(node, libcst.ClassDef)
        and node.name.value == model_skeleton.ProcessorModel.__name__
    )

    load_def = libcst.parse_statement(
        f"""
def load(self) -> None:
    self._processor = {processor_desrciptor.cls_name}(context=self._context)
"""
    )

    endpoint_descriptor = processor_desrciptor.endpoint
    def_str = "async def" if endpoint_descriptor.is_async else "def"
    # Convert json payload dict to processor args.
    obj_arg_parts = ", ".join(
        (
            f"{arg_name}={arg_type.as_src_str()}.parse_obj(payload['{arg_name}'])"
            if arg_type.is_pydantic
            else f"{arg_name}=payload['{arg_name}']"
        )
        for arg_name, arg_type in endpoint_descriptor.input_names_and_tyes
    )

    if len(endpoint_descriptor.output_types) == 1:
        output_type = endpoint_descriptor.output_types[0]
        result = "result.dict()" if output_type.is_pydantic else "result"
    else:
        result_parts = [
            f"result[{i}].dict()" if t.is_pydantic else f"result[{i}]"
            for i, t in enumerate(endpoint_descriptor.output_types)
        ]
        result = f"({', '.join(result_parts)})"

    maybe_await = "await " if endpoint_descriptor.is_async else ""

    predict_def = libcst.parse_statement(
        f"""
{def_str} predict(self, payload):
    result = {maybe_await}self._processor.{endpoint_descriptor.name}({obj_arg_parts})
    return  {result}

"""
    )
    new_body: list[libcst.BaseStatement] = list(  # type: ignore[assignment,misc]
        class_definition.body.body
    ) + [
        load_def,
        predict_def,
    ]
    new_block = libcst.IndentedBlock(body=new_body)
    class_definition = class_definition.with_changes(body=new_block)
    class_definition = class_definition.visit(  # type: ignore[assignment]
        _SpecifyProcessorTypeAnnotation(processor_desrciptor.cls_name)
    )

    if issubclass(processor_desrciptor.user_config_type.raw, type(None)):
        userconfig_pin = libcst.parse_statement("UserConfigT = None")
    else:
        userconfig_pin = libcst.parse_statement(
            f"UserConfigT = {processor_desrciptor.user_config_type.as_src_str()}"
        )

    return class_definition, imports, userconfig_pin

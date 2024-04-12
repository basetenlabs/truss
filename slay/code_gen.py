import ast
import json
import logging
import pathlib
import shlex
import subprocess
import textwrap
from typing import Any, Iterable, Mapping, Optional, cast

import datamodel_code_generator
import libcst
from datamodel_code_generator import model as generator_models
from datamodel_code_generator.parser import jsonschema as jsonschema_parser
from slay import definitions, utils
from slay.truss_adapter import code_gen

INDENT = " " * 4

_DEFS_KEY = "$defs"


def _indent(text: str, num: int = 1) -> str:
    return textwrap.indent(text, INDENT * num)


class _MainRemover(ast.NodeTransformer):
    """Removes main-section from module AST."""

    def visit_If(self, node):
        """Robustly matches variations of `if __name__ == "__main__":`."""
        if (
            isinstance(node.test, ast.Compare)
            and any(
                isinstance(c, ast.Name) and c.id == "__name__"
                for c in ast.walk(node.test.left)
            )
            and any(
                isinstance(c, ast.Constant) and c.value == "__main__"
                for c in ast.walk(node.test)
            )
        ):
            return None
        return self.generic_visit(node)


def _remove_main_section(source_code: str) -> str:
    """Removes main-section from module source."""
    parsed_code = ast.parse(source_code)
    transformer = _MainRemover()
    transformed_ast = transformer.visit(parsed_code)
    return ast.unparse(transformed_ast)


def _run_simple_subprocess(cmd: str) -> None:
    process = subprocess.Popen(
        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    _, stderr = process.communicate()
    if process.returncode != 0:
        raise ChildProcessError(f"Error: {stderr.decode()}")


def _format_python_file(file_path: pathlib.Path) -> None:
    _run_simple_subprocess(
        f"autoflake --in-place --remove-all-unused-imports {file_path}"
    )
    _run_simple_subprocess(f"black {file_path}")
    _run_simple_subprocess(f"isort {file_path}")


def make_processor_dir(
    workflow_root: pathlib.Path,
    workflow_name: str,
    processor_descriptor: definitions.ProcessorAPIDescriptor,
) -> pathlib.Path:
    processor_name = processor_descriptor.name
    file_name = f"processor_{processor_name}"
    processor_dir = (
        workflow_root / definitions.GENERATED_CODE_DIR / workflow_name / file_name
    )
    processor_dir.mkdir(exist_ok=True, parents=True)
    return processor_dir


# Stub Gen #############################################################################


def _endpoint_signature_src(endpoint: definitions.EndpointAPIDescriptor):
    """
    E.g.: `async def run(self, data: str, num_partitions: int) -> tuple[list, int]:`
    """
    if endpoint.is_generator:
        # TODO: implement generator.
        raise NotImplementedError("Generator.")

    def_str = "async def" if endpoint.is_async else "def"
    args = ", ".join(
        f"{arg_name}: {arg_type.as_src_str(definitions.STUB_TYPE_MODULE)}"
        for arg_name, arg_type in endpoint.input_names_and_types
    )
    if len(endpoint.output_types) == 1:
        output_type = (
            f"{endpoint.output_types[0].as_src_str(definitions.STUB_TYPE_MODULE)}"
        )
    else:
        out_types = ", ".join(
            t.as_src_str(definitions.STUB_TYPE_MODULE) for t in endpoint.output_types
        )
        output_type = f"tuple[{out_types}]"
    return f"{def_str} {endpoint.name}(self, {args}) -> {output_type}:"


def _endpoint_body_src(endpoint: definitions.EndpointAPIDescriptor) -> str:
    """Generates source code for calling the stub and wrapping the I/O types.

    E.g.:
    ```
    json_args = {"inputs": inputs.dict(), "extra_arg": extra_arg}
    json_result = await self._remote.predict_async(json_args)
    return (SplitTextOutput.parse_obj(json_result[0]), json_result[1])
    ```
    """
    if endpoint.is_generator:
        raise NotImplementedError("Generator")

    parts = []
    # Pack arg list as json dictionary.
    json_arg_parts = (
        (
            f"'{arg_name}': {arg_name}.dict()"
            if arg_type.is_pydantic
            else f"'{arg_name}': {arg_name}"
        )
        for arg_name, arg_type in endpoint.input_names_and_types
    )
    json_args = f"{{{', '.join(json_arg_parts)}}}"
    parts.append(f"json_args = {json_args}")
    # Invoke remote.
    remote_call = (
        "await self._remote.predict_async(json_args)"
        if endpoint.is_async
        else "self._remote.predict_sync(json_args)"
    )
    parts.append(f"json_result = {remote_call}")
    # Unpack response and parse as pydantic models if needed.
    if len(endpoint.output_types) == 1:
        output_type = utils.expect_one(endpoint.output_types)
        if output_type.is_pydantic:
            type_str = output_type.as_src_str(definitions.STUB_TYPE_MODULE)
            ret = f"{type_str}.parse_obj(json_result)"
        else:
            ret = "json_result"
    else:
        ret_parts = ", ".join(
            (
                (
                    f"{output_type.as_src_str(definitions.STUB_TYPE_MODULE)}"
                    f".parse_obj(json_result[{i}])"
                )
                if output_type.is_pydantic
                else f"json_result[{i}]"
            )
            for i, output_type in enumerate(endpoint.output_types)
        )
        ret = f"{ret_parts}"
    parts.append(f"return {ret}")

    return "\n".join(parts)


def _gen_stub_src(
    processor: definitions.ProcessorAPIDescriptor,
    maybe_pydantic_types_file: Optional[pathlib.Path],
) -> tuple[str, list[str]]:
    """Generates stub class source, e.g:

    ```
    from slay import stub

    class SplitText(stub.StubBase):
        def __init__(self, url: str, api_key: str) -> None:
            self._remote = stub.BasetenSession(url, api_key)

        async def run(self, data: str, num_partitions: int) -> tuple[SplitTextOutput, int]:
            json_args = {"inputs": inputs.dict(), "extra_arg": extra_arg}
            json_result = await self._remote.predict_async(json_args)
            return (SplitTextOutput.parse_obj(json_result[0]), json_result[1])
    ```
    """
    imports = ["from slay import stub"]
    if maybe_pydantic_types_file:
        imports.append(f"import {definitions.STUB_TYPE_MODULE}")

    src_parts = [
        f"class {utils.make_stub_name(processor.name)}(stub.StubBase):",
        _indent(_endpoint_signature_src(processor.endpoint)),
        _indent(_endpoint_body_src(processor.endpoint), 2),
        "\n\n",
    ]
    return "\n".join(src_parts), imports


def _remove_root_model(pydantic_src: str) -> str:
    """To process source code generated by `datamodel_code_generator`."""
    tree = ast.parse(pydantic_src)
    src = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            continue
        else:
            src.append(node)

    return f"{ast.unparse(ast.Module(body=src, type_ignores=[]))}\n\n"


def _export_pydantic_schemas(
    dependencies: Iterable[definitions.ProcessorAPIDescriptor],
) -> Mapping[str, Any]:
    """Creates a dict with all pydantic schemas used as input or output types by the
    dependencies.

    If a schema itself depends on another class (e.g. an enum definition), these
    dependent schemas are also added to the dict.

    It is enforced that pydantic models across the code base have different names,
    to avoid conflicts; automatic disambiguation is not yet implemented.
    """
    name_to_schema: dict[str, Any] = {}

    def safe_add_schema(type_descr: definitions.TypeDescriptor):
        if not type_descr.is_pydantic:
            return

        name = type_descr.raw.__name__
        schema = type_descr.raw.schema()
        if existing_schema := name_to_schema.get(name):
            if existing_schema != schema:
                raise NotImplementedError(
                    f"Two pydantic models with same name `{name}."
                )
        else:
            # Move definitions needed by the current type to top level.
            if _DEFS_KEY in schema:
                for def_name, def_schema in schema[_DEFS_KEY].items():
                    if existing_def_schema := name_to_schema.get(def_name):
                        if existing_def_schema != def_schema:
                            raise NotImplementedError(
                                f"Two pydantic models with same name `{def_name}."
                            )
                    else:
                        name_to_schema[def_name] = def_schema
                schema.pop(_DEFS_KEY)

            name_to_schema[name] = schema

    for dep in dependencies:
        for _, input_type in dep.endpoint.input_names_and_types:
            safe_add_schema(input_type)
        for output_type in dep.endpoint.output_types:
            safe_add_schema(output_type)
    return name_to_schema


def gen_pydantic_models(
    file_path: pathlib.Path,
    dependencies: Iterable[definitions.ProcessorAPIDescriptor],
) -> Optional[pathlib.Path]:
    name_to_schema = _export_pydantic_schemas(dependencies)

    if not name_to_schema:
        return None

    combined_schema = {_DEFS_KEY: name_to_schema}
    # TODO: parameterize pydantic and python version properly.
    py_version = datamodel_code_generator.PythonVersion.PY_39
    data_model_types = generator_models.get_data_model_types(
        data_model_type=datamodel_code_generator.DataModelType.PydanticBaseModel,
        target_python_version=py_version,
    )
    with utils.log_level(logging.INFO):
        parser = jsonschema_parser.JsonSchemaParser(
            json.dumps(combined_schema),
            data_model_type=data_model_types.data_model,
            data_model_root_type=data_model_types.root_model,
            data_model_field_type=data_model_types.field_model,
            data_type_manager_type=data_model_types.data_type_manager,
            target_python_version=py_version,
            capitalise_enum_members=True,
            use_subclass_enum=True,
            strict_nullable=True,
            apply_default_values_for_required_fields=True,
            use_default_kwarg=True,
            use_standard_collections=True,
        )
        pydantic_module_src: str = cast(str, parser.parse())
    file_path.write_text(_remove_root_model(pydantic_module_src))
    return file_path


def _generate_stub_src_for_deps(
    dependencies: Iterable[definitions.ProcessorAPIDescriptor],
    pydantic_types_file: Optional[pathlib.Path],
) -> Optional[tuple[str, set[str]]]:
    """Generates a source code and imports for stub classes."""
    imports = set()
    src_parts = []

    for dep in dependencies:
        stub_src, new_imports = _gen_stub_src(dep, pydantic_types_file)
        imports.update(new_imports)
        src_parts.append(stub_src)

    if not (imports or src_parts):
        return None
    return "\n".join(src_parts), imports


# Remote Processor Gen #################################################################


def generate_processor_source(
    file_path: pathlib.Path,
    processor_descriptor: definitions.ProcessorAPIDescriptor,
    dependencies: Iterable[definitions.ProcessorAPIDescriptor],
    pydantic_types_file: Optional[pathlib.Path],
):
    """Generates code that wraps a processor as a truss-compatible model."""
    maybe_stub_src = _generate_stub_src_for_deps(dependencies, pydantic_types_file)

    source_code = _remove_main_section(file_path.read_text())
    if maybe_stub_src:
        stub_src, stub_imports = maybe_stub_src
        stub_imports_str = "\n".join(stub_imports)
        source_code = f"{stub_imports_str}\n{source_code}\n{stub_src}"

    source_tree = libcst.parse_module(source_code)

    # TODO: Processor isolation: either prune file from high-confidence unneeded.

    model_def, imports, userconfig_pin = code_gen.generate_truss_model(
        processor_descriptor
    )
    new_body: list[Any] = imports + list(source_tree.body) + [userconfig_pin, model_def]
    source_tree = source_tree.with_changes(body=new_body)
    file_path.write_text(source_tree.code)
    _format_python_file(file_path)

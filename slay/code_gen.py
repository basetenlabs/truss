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

STUB_MODULE = "remote_stubs"
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
    processor_name = processor_descriptor.cls_name
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
        f"{arg_name}: {arg_type.as_src_str()}"
        for arg_name, arg_type in endpoint.input_names_and_types
    )
    if len(endpoint.output_types) == 1:
        output_type = f"{endpoint.output_types[0].as_src_str()}"
    else:
        output_type = (
            f"tuple[{', '.join(t.as_src_str() for t in endpoint.output_types)}]"
        )
    return f"{def_str} {endpoint.name}(self, {args}) -> {output_type}:"


def _gen_protocol_src(
    processor: definitions.ProcessorAPIDescriptor,
) -> tuple[str, list[str]]:
    """Generates source code for a Protocol that matches the processor."""
    imports = ["from typing import Protocol"]
    src_parts = [
        f"class {processor.cls_name}P(Protocol):",
        _indent(f"{_endpoint_signature_src(processor.endpoint)})"),
        _indent("...", 2),
    ]
    return "\n".join(src_parts), imports


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
            ret = f"{output_type.as_src_str()}.parse_obj(json_result)"
        else:
            ret = "json_result"
    else:
        ret_parts = ", ".join(
            (
                f"{output_type.as_src_str()}.parse_obj(json_result[{i}])"
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

    src_parts = [
        f"class {processor.cls_name}(stub.StubBase):",
        _indent(_endpoint_signature_src(processor.endpoint)),
        _indent(_endpoint_body_src(processor.endpoint), 2),
        "\n\n",
    ]
    return "\n".join(src_parts), imports


def _remove_root_model_and_separate_imports(pydantic_src: str) -> tuple[str, list[str]]:
    """To process source code generated by `datamodel_code_generator`."""
    tree = ast.parse(pydantic_src)
    imports = []
    other_src = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
        elif isinstance(node, ast.ClassDef) and node.name == "Model":
            continue
        else:
            other_src.append(node)

    return f"{ast.unparse(ast.Module(body=other_src, type_ignores=[]))}\n\n", imports


def _export_pydantic_schemas(
    dependencies: Iterable[definitions.ProcessorAPIDescriptor],
) -> Mapping[str, Any]:
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


def _gen_pydantic_models(
    dependencies: Iterable[definitions.ProcessorAPIDescriptor],
) -> tuple[str, list[str]]:
    name_to_schema = _export_pydantic_schemas(dependencies)

    if not name_to_schema:
        return "", []

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
    return _remove_root_model_and_separate_imports(pydantic_module_src)


def generate_stubs_for_deps(
    processor_dir: pathlib.Path,
    dependencies: Iterable[definitions.ProcessorAPIDescriptor],
) -> Optional[pathlib.Path]:
    """Generates a source file with stub classes."""
    imports = set()
    src_parts = []

    pydantic_src, pydantic_imports = _gen_pydantic_models(dependencies)
    if pydantic_src:
        src_parts.append(pydantic_src)
        imports.update(pydantic_imports)

    for dep in dependencies:
        # protocol_src, new_deps = gen_protocol_src(dep)
        # imports.update(new_deps)
        # src_parts.append(protocol_src)
        stub_src, new_deps = _gen_stub_src(dep)
        imports.update(new_deps)
        src_parts.append(stub_src)

    if not (imports or src_parts):
        return None

    out_file_path = processor_dir / f"{STUB_MODULE}.py"
    with out_file_path.open("w") as fp:
        fp.write("\n".join(imports))
        fp.write("\n")
        fp.writelines(src_parts)

    _format_python_file(out_file_path)
    return out_file_path


# Remote Processor Gen #################################################################


class _InitRewriter(libcst.CSTTransformer):
    """Removes processors from init args and instead initializes corresponding stubs."""

    def __init__(self, cls_name: str, replacements):
        super().__init__()
        self._cls_name = cls_name
        self._replacements = replacements

    def leave_ClassDef(
        self, original_node: libcst.ClassDef, updated_node: libcst.ClassDef
    ) -> libcst.ClassDef:
        # Target only the Workflow class
        if original_node.name.value != self._cls_name:
            return updated_node

        new_methods: list[Any] = []
        for method in updated_node.body.body:
            if (
                isinstance(method, libcst.FunctionDef)
                and method.name.value == "__init__"
            ):
                new_method = self._modify_init_method(method)
                new_methods.append(new_method)
            else:
                new_methods.append(method)
        return updated_node.with_changes(
            body=updated_node.body.with_changes(body=new_methods)
        )

    def _modify_init_method(self, method: libcst.FunctionDef) -> libcst.FunctionDef:
        keep_params_names = {definitions.SELF_ARG_NAME, definitions.CONTEXT_ARG_NAME}
        if method.name.value == "__init__":
            # Drop other params - assumes that we have verified that all arguments
            # are processors.
            keep_params = []
            for param in method.params.params:
                if param.name.value in keep_params_names:
                    keep_params.append(param)
                else:
                    if param.name.value not in self._replacements:
                        raise ValueError(
                            f"For argument `{param.name.value}` no processor was "
                            f"mapped. Available {list(self._replacements.keys())}"
                        )

            new_params = method.params.with_changes(params=keep_params)

            processor_assignments = [
                libcst.parse_statement(
                    f"{name} = stub.stub_factory({stub_cls_ref}, context)"
                )
                for name, stub_cls_ref in self._replacements.items()
            ]

            # Create new statements for the method body
            new_body = method.body.with_changes(
                body=processor_assignments + list(method.body.body)
            )

            return method.with_changes(params=new_params, body=new_body)
        return method


def _rewrite_processor_inits(
    source_tree: libcst.Module, processor_descriptor: definitions.ProcessorAPIDescriptor
):
    """Removes processors from init args and instead initializes corresponding stubs."""
    replacements = {}
    for name, proc_cls in processor_descriptor.dependencies.items():
        replacements[name] = f"{STUB_MODULE}.{proc_cls.__name__}"

    if not replacements:
        return source_tree

    modified_tree = source_tree.visit(
        _InitRewriter(processor_descriptor.cls_name, replacements)
    )

    new_imports = [
        libcst.parse_statement(f"import {STUB_MODULE}"),
        libcst.parse_statement("from slay import stub"),
    ]

    modified_tree = modified_tree.with_changes(
        body=new_imports + list(modified_tree.body)
    )
    return modified_tree


########################################################################################


def generate_processor_source(
    file_path: pathlib.Path,
    processor_descriptor: definitions.ProcessorAPIDescriptor,
):
    """Generates code that wraps a processor as a truss-compatible model."""
    source_code = _remove_main_section(file_path.read_text())
    source_tree = libcst.parse_module(source_code)
    source_tree = _rewrite_processor_inits(source_tree, processor_descriptor)

    # TODO: Processor isolation: either prune file or generate a new file.
    #   At least remove main section.

    model_def, imports, userconfig_pin = code_gen.generate_truss_model(
        processor_descriptor
    )
    new_body: list[Any] = imports + list(source_tree.body) + [userconfig_pin, model_def]
    source_tree = source_tree.with_changes(body=new_body)
    file_path.write_text(source_tree.code)
    _format_python_file(file_path)

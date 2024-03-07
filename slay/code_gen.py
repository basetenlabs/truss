import logging
import os
import pathlib
import shutil
import textwrap
from typing import Iterable

import black
import isort
import libcst
import libcst as cst
from rope.base import project as rope_project
from rope.refactor import move as rope_move
from rope.refactor import occurrences as rope_occurrences
from slay import definitions, truss_model_skeleton, utils
from truss import truss_config

INDENT = " " * 4


def _indent(text: str) -> str:
    return textwrap.indent(text, INDENT)


def _format_python_file(file_path):
    with utils.log_level(logging.INFO):
        print(f"formating {file_path}")
        black.format_file_in_place(
            pathlib.Path(file_path), fast=False, mode=black.FileMode()
        )
    with utils.no_print():
        isort.file(file_path)


def _read_source(file_path):
    with open(file_path, "r", encoding="utf-8") as source_file:
        source_code = source_file.read()
    return source_code


# Stub Gen #############################################################################


def _endpoint_signature_src(endpoint: definitions.EndpointAPIDescriptor):
    if endpoint.is_generator:
        # TODO: implement generator.
        raise NotImplementedError("Generator")

    def_str = "async def" if endpoint.is_async else "def"
    args = ", ".join(
        f"{arg_name}: {arg_type.as_str()}"
        for arg_name, arg_type in endpoint.input_name_and_tyes
    )
    if len(endpoint.output_types) == 1:
        output_type = f"{endpoint.output_types[0].as_str()}"
    else:
        output_type = f"tuple[{', '.join(t.as_str() for t in endpoint.output_types)}]"
    return f"""{def_str} {endpoint.name}(self, {args}) -> {output_type}:"""


def gen_protocol_src(processor: definitions.ProcessorAPIDescriptor):
    # TODO: Add pydantic type definitions/imports.
    imports = ["from typing import Protocol"]
    src_parts = [
        f"""
class {processor.cls_name}P(Protocol):
"""
    ]
    for endpoint in processor.endpoints:
        src_parts.append(_indent(f"{_endpoint_signature_src(endpoint)}\n{INDENT}...\n"))
    return "\n".join(src_parts), imports


def endpoint_body_src(endpoint: definitions.EndpointAPIDescriptor):
    if endpoint.is_generator:
        raise NotImplementedError("Generator")

    json_arg_parts = (
        (
            f"'{arg_name}': {arg_name}.model_dump()"
            if arg_type.is_pydantic
            else f"'{arg_name}': {arg_name}"
        )
        for arg_name, arg_type in endpoint.input_name_and_tyes
    )

    json_args = f"{{{', '.join(json_arg_parts)}}}"
    remote_call = (
        "await self._remote.predict_async(json_args)"
        if endpoint.is_async
        else "self._remote.predict_sync(json_args)"
    )

    if len(endpoint.output_types) == 1:
        ret = "json_result"
    else:
        ret_parts = ", ".join(
            (
                f"{output_type.as_str()}.model_validate(json_result[{i}])"
                if output_type.is_pydantic
                else f"json_result[{i}]"
            )
            for i, output_type in enumerate(endpoint.output_types)
        )
        ret = f"({ret_parts})"

    body = f"""
json_args = {json_args}
json_result = {remote_call}
return {ret}
"""
    return body


def gen_stub_src(processor: definitions.ProcessorAPIDescriptor):
    # TODO: Add pydantic type definitions/imports.
    imports = ["from slay import stub"]

    src_parts = [
        f"""
class {processor.cls_name}(stub.StubBase):

    def __init__(self, url: str, api_key: str) -> None:
        self._remote = stub.BasetenSession(url, api_key)
"""
    ]
    for endpoint in processor.endpoints:
        body = _indent(endpoint_body_src(endpoint))
        src_parts.append(
            _indent(
                f"{_endpoint_signature_src(endpoint)}{body}\n",
            )
        )
    return "\n".join(src_parts), imports


def generate_stubs_for_deps(
    out_file_path, dependencies: Iterable[definitions.ProcessorAPIDescriptor]
):
    imports = set()
    src_parts = []
    for dep in dependencies:
        # protocol_src, new_deps = gen_protocol_src(dep)
        # imports.update(new_deps)
        # src_parts.append(protocol_src)
        stub_src, new_deps = gen_stub_src(dep)
        imports.update(new_deps)
        src_parts.append(stub_src)

    if not (imports or src_parts):
        return

    with open(out_file_path, "w", encoding="utf-8") as fp:
        fp.writelines("\n".join(imports))
        fp.writelines(src_parts)

    _format_python_file(out_file_path)


# Remote Model Gen #####################################################################


class InitRewriter(libcst.CSTTransformer):
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

        new_methods = []
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
        keep_params_names = ["self", "context"]  # TODO: introduce constants.
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
                            f"mappend. Available {list(self._replacements.keys())}"
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


def rewrite_processor_inits(
    source_tree: libcst.Module, processor_desrciptor: definitions.ProcessorAPIDescriptor
):
    replacements = {}
    for name, proc_cls in processor_desrciptor.depdendencies.items():
        replacements[name] = f"user_stubs.{proc_cls.__name__}"

    if not replacements:
        return source_tree

    logging.info(f"Adding stub inits to `{processor_desrciptor.cls_name}`.")

    modified_tree = source_tree.visit(
        InitRewriter(processor_desrciptor.cls_name, replacements)
    )

    new_imports = [
        libcst.parse_statement(f"from . import user_stubs"),
        libcst.parse_statement(f"from slay import stub"),
    ]

    modified_tree = modified_tree.with_changes(
        body=new_imports + list(modified_tree.body)
    )
    return modified_tree


def _rope_find_def_offset(project, source_module, symbol_name):
    finder = rope_occurrences.Finder(project, symbol_name)
    offset = -1
    occurrences = finder.find_occurrences(source_module)
    for occurrence in occurrences:
        if occurrence.is_defined():
            if offset >= 0:
                raise ValueError("Multiple found")
            offset = occurrence.offset
    if offset < 0:
        raise ValueError("Not found")
    return offset


def move_class_to_new_file(
    project_root,
    source_file_path,
    target_file_path,
    class_name,
):
    logging.info(f"Moving`{class_name}` dedicated model file.")
    # TODO: all of this is totally unclear.
    project = rope_project.Project(project_root, ropefolder=None)
    source_module = project.get_resource(source_file_path)

    offset = _rope_find_def_offset(project, source_module, class_name)

    target_file = project.get_file(target_file_path)
    if target_file.exists():
        target_file.remove()
    target_file.create()

    target_resource = project.get_resource(target_file_path)

    print(f"Moving `{class_name}` to `{target_file_path}`.")
    move_refactoring = rope_move.create_move(project, source_module, offset)

    changes = move_refactoring.get_changes(dest=target_resource)
    # TODO: move other needed stuff from source module - so we can get rid of it?
    project.do(changes)


class ChangeProcessorAnnotation(cst.CSTTransformer):
    def __init__(self, new_annotaiton: str) -> None:
        super().__init__()
        self._new_annotaiton = new_annotaiton

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> cst.SimpleStatementLine:
        new_body = []
        for statement in updated_node.body:
            if (
                isinstance(statement, cst.AnnAssign)
                and isinstance(statement.target, cst.Name)
                and statement.target.value == "_processor"
            ):
                new_annotation = cst.Annotation(
                    annotation=cst.Name(value=self._new_annotaiton)
                )
                new_statement = statement.with_changes(annotation=new_annotation)
                new_body.append(new_statement)
            else:
                new_body.append(statement)

        return updated_node.with_changes(body=tuple(new_body))


def generate_baseten_model(processor_desrciptor: definitions.ProcessorAPIDescriptor):
    logging.info(f"Generating Baseten model for `{processor_desrciptor.cls_name}`.")
    remote_tree = libcst.parse_module(_read_source(truss_model_skeleton.__file__))

    new_imports = [
        node
        for node in remote_tree.body
        if isinstance(node, libcst.SimpleStatementLine)
        and any(
            isinstance(stmt, libcst.Import) or isinstance(stmt, libcst.ImportFrom)
            for stmt in node.body
        )
    ]

    class_definition = utils.expect_one(
        node
        for node in remote_tree.body
        if isinstance(node, libcst.ClassDef) and node.name.value == "Model"
    )

    load_def = libcst.parse_statement(
        f"""
def load(self) -> None:
    self._processor = {processor_desrciptor.cls_name}(context=self._context)
"""
    )

    endpoint_descriptor = utils.expect_one(processor_desrciptor.endpoints)
    def_str = "async def" if endpoint_descriptor.is_async else "def"
    # Convert json payload dict to processor args.
    obj_arg_parts = ", ".join(
        (
            f"{arg_name}={arg_type.as_str()}.model_validate(payload['{arg_name}'])"
            if arg_type.is_pydantic
            else f"{arg_name}=payload['{arg_name}']"
        )
        for arg_name, arg_type in endpoint_descriptor.input_name_and_tyes
    )

    if len(endpoint_descriptor.output_types) == 1:
        output_type = endpoint_descriptor.output_types[0]
        result = "result.model_dump()" if output_type.is_pydantic else "result"
    else:
        result_parts = [
            f"result[{i}].model_dump()" if t.is_pydantic else f"result[{i}]"
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

    new_block = libcst.IndentedBlock(
        body=list(class_definition.body.body) + [load_def, predict_def]
    )
    class_definition = class_definition.with_changes(body=new_block)
    class_definition = class_definition.visit(
        ChangeProcessorAnnotation(processor_desrciptor.cls_name)
    )
    return class_definition, new_imports


########################################################################################


def edit_model_source(
    file_path,
    processor_desrciptor: definitions.ProcessorAPIDescriptor,
):
    wdir = os.path.dirname(file_path)
    source_code = _read_source(file_path)
    source_tree = libcst.parse_module(source_code)
    source_tree = rewrite_processor_inits(source_tree, processor_desrciptor)
    modified_source_code = source_tree.code

    with open(file_path, "w", encoding="utf-8") as modified_file:
        modified_file.write(modified_source_code)

    model_file_path = os.path.join(wdir, "model.py")
    move_class_to_new_file(
        wdir,
        os.path.basename(file_path),
        os.path.basename(model_file_path),
        processor_desrciptor.cls_name,
    )
    # Restore source file to pre move.
    with open(file_path, "w", encoding="utf-8") as modified_file:
        modified_file.write(modified_source_code)

    model_source_code = _read_source(model_file_path)

    model_source_tree = libcst.parse_module(model_source_code)
    model_def, imports = generate_baseten_model(processor_desrciptor)
    model_source_tree = model_source_tree.with_changes(
        body=imports + list(model_source_tree.body) + [model_def]
    )
    with open(model_file_path, "w", encoding="utf-8") as modified_file:
        modified_file.write(model_source_tree.code)

    _format_python_file(model_file_path)


def make_truss_dir(
    processor_dir,
    processor_desrciptor: definitions.ProcessorAPIDescriptor,
    stub_cls_to_url,
):
    # TODO: Handle if model uses truss config instead of `defautl_config`.
    # TODO: Handle file-based overrides when deploying.
    default_config = processor_desrciptor.processor_cls.default_config
    config = truss_config.TrussConfig()
    config.model_name = default_config.name or processor_desrciptor.cls_name
    config.resources.cpu = "1"
    config.resources.use_gpu = False
    config.secrets = {"baseten_api_key": "BASETEN_API_KEY"}
    config.python_version = "3.11"
    config.base_image = truss_config.BaseImage(image="python:3.11-slim")
    config.requirements_file = "requirements.txt"

    slay_config = definitions.TrussMetadata(
        user_config=processor_desrciptor.processor_cls.default_config.user_config,
        stub_cls_to_url=stub_cls_to_url,
    )

    config.model_metadata["slay_metadata"] = slay_config.model_dump()

    truss_dir = processor_dir / "truss"
    truss_dir.mkdir(exist_ok=True)

    config.write_to_yaml_file(truss_dir / "config.yaml", verbose=False)

    # Copy other sources.
    shutil.copy(processor_dir / "requirements.txt", truss_dir)

    model_dir = truss_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(processor_dir / "model.py", model_dir)

    pkg_dir = truss_dir / "packages"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    for module in processor_dir.glob("*.py"):
        if module.name != "model.py":
            shutil.copy(module, pkg_dir)

    # This dependency should be handled automatically.
    # Also: apparently packages need an `__init__`, or crash.
    shutil.copytree(
        "/home/marius-baseten/workbench/truss/example_workflow/user_package",
        pkg_dir / "user_package",
        dirs_exist_ok=True,
    )

    # This should be included in truss or a lib, not manually copied.
    shutil.copytree(
        "/home/marius-baseten/workbench/truss/slay",
        pkg_dir / "slay",
        dirs_exist_ok=True,
    )

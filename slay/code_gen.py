import logging
import os
import pathlib
import shutil
import textwrap
from typing import Iterable

import black
import isort
import libcst
import pydantic
from rope.base import project as rope_project
from rope.refactor import move as rope_move
from rope.refactor import occurrences as rope_occurrences
from slay import definitions, utils

INDENT = " " * 4


def _indent(text: str) -> str:
    return textwrap.indent(text, INDENT)


def _format_python_file(file_path):
    with utils.log_level(logging.INFO):
        black.format_file_in_place(
            pathlib.Path(file_path), fast=False, mode=black.FileMode()
        )
    with utils.no_print():
        isort.file(file_path)


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
class {processor.processor_cls.__name__}P(Protocol):
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
            f"{arg_name}.model_dump_json()"
            if issubclass(arg_type.raw, pydantic.BaseModel)
            else arg_name
        )
        for arg_name, arg_type in endpoint.input_name_and_tyes
    )

    json_args = f"[{', '.join(json_arg_parts)}]"
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
                if issubclass(output_type.raw, pydantic.BaseModel)
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
class {processor.processor_cls.__name__}(stub.StubBase):

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
        protocol_src, new_deps = gen_protocol_src(dep)
        imports.update(new_deps)
        src_parts.append(protocol_src)
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
                            f"For argument `{param.name.value}` no processor was mappend. Available {list(self._replacements.keys())}"
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
        replacements[name] = f"dependencies.{proc_cls.__name__}"

    logging.info(f"Rewriting `{processor_desrciptor.processor_cls.__name__}`.")

    modified_tree = source_tree.visit(
        InitRewriter(processor_desrciptor.processor_cls.__name__, replacements)
    )

    new_imports = [
        libcst.parse_statement(f"from . import dependencies"),
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
    # TODO: move other needed stuff from source module.
    project.do(changes)


########################################################################################


def modify_source_file(
    file_path,
    processor_desrciptor: definitions.ProcessorAPIDescriptor,
):
    wdir = os.path.dirname(file_path)

    with open(file_path, "r", encoding="utf-8") as source_file:
        source_code = source_file.read()

    source_tree = libcst.parse_module(source_code)
    source_tree = rewrite_processor_inits(source_tree, processor_desrciptor)
    modified_source_code = source_tree.code

    with open(file_path, "w", encoding="utf-8") as modified_file:
        modified_file.write(modified_source_code)

    target_file_path = os.path.join(wdir, "model.py")
    move_class_to_new_file(
        wdir,
        os.path.basename(file_path),
        os.path.basename(target_file_path),
        processor_desrciptor.processor_cls.__name__,
    )

    _format_python_file(file_path)

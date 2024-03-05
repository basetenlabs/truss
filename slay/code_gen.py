import logging
import pathlib
import textwrap
from typing import Iterable

import black
import isort
import libcst
import pydantic
from slay import definitions, utils

INDENT = " " * 4


def _indent(text: str) -> str:
    return textwrap.indent(text, INDENT)


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

    _clean_python_file(out_file_path)


# Remote Model Gen #####################################################################


########################################################################################


def _clean_python_file(file_path):
    with utils.log_level(logging.INFO):
        black.format_file_in_place(
            pathlib.Path(file_path), fast=False, mode=black.FileMode()
        )
    with utils.no_print():
        isort.file(file_path)


def modify_source_file(file_path, class_name: str):
    with open(file_path, "r", encoding="utf-8") as source_file:
        source_code = source_file.read()

    modified_source_code = source_code

    with open(file_path, "w", encoding="utf-8") as modified_file:
        modified_file.write(modified_source_code)

    _clean_python_file(file_path)

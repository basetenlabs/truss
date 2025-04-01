"""
Chains currently assumes that everything from the directory in which the entrypoint
is defined (i.e. sibling files and nested dirs) could be imported/used. e.g.:

workspace/
  entrypoint.py
  helper.py
  some_package/
    utils.py
    sub_package/
      ...

These sources are copied into truss's `/packages` and can be imported on the remote.
Using code *outside* of the workspace is not supported:

shared_lib/
  common.py
workspace/
  entrypoint.py
  ...

`shared_lib` can only be imported on the remote if its installed as a pip
requirement (site-package), it will not be copied from the local host.
"""

import logging
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Iterable, Mapping, Optional, cast, get_args, get_origin

import libcst
import pydantic

import truss
from truss.base import custom_types, truss_config
from truss.contexts.image_builder import serving_image_builder
from truss.util import path as truss_path
from truss_chains import framework, private_types, public_types, utils

_INDENT = " " * 4
_REQUIREMENTS_FILENAME = "pip_requirements.txt"
_MODEL_FILENAME = "model.py"
_MODEL_CLS_NAME = "TrussChainletModel"
_TRUSS_GIT = "git+https://github.com/basetenlabs/truss.git"
_TRUSS_PIP_PATTERN = re.compile(
    r"""
    ^truss
    (?:
        \s*(==|>=|<=|!=|>|<)\s*   # Version comparison operators
        \d+(\.\d+)*               # Version numbers (e.g., 1, 1.0, 1.0.0)
        (?:                       # Optional pre-release or build metadata
            (?:a|b|rc|dev)\d*
            (?:\.post\d+)?
            (?:\+[\w\.]+)?
        )?
    )?$
""",
    re.VERBOSE,
)

_MODEL_SKELETON_FILE = (
    pathlib.Path(__file__).parent.parent.resolve()
    / "remote_chainlet"
    / "model_skeleton.py"
)


def _indent(text: str, num: int = 1) -> str:
    return textwrap.indent(text, _INDENT * num)


def _run_simple_subprocess(cmd: str) -> None:
    process = subprocess.Popen(
        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    _, stderr = process.communicate()
    if process.returncode != 0:
        raise ChildProcessError(f"Error: {stderr.decode()}")


def _format_python_file(file_path: pathlib.Path) -> None:
    # Resolve importing sorting and unused import issues.
    _run_simple_subprocess(f"ruff check {file_path} --fix --select F401,I")
    _run_simple_subprocess(f"ruff format {file_path}")


class _Source(custom_types.SafeModelNonSerializable):
    src: str
    imports: set[str] = pydantic.Field(default_factory=set)


def _update_src(new_source: _Source, src_parts: list[str], imports: set[str]) -> None:
    src_parts.append(new_source.src)
    imports.update(new_source.imports)


def _gen_pydantic_import_and_ref(raw_type: Any) -> _Source:
    """Returns e.g. ("from sub_package import module", "module.OutputType")."""
    if raw_type.__module__ == "__main__":
        # Assuming that main is copied into package dir and can be imported.
        module_obj = sys.modules[raw_type.__module__]
        if not module_obj.__file__:
            raise public_types.ChainsUsageError(
                f"File-based python code required. `{raw_type}` does not have a file."
            )

        file = os.path.basename(module_obj.__file__)
        assert file.endswith(".py")
        module_name = file.replace(".py", "")
        import_src = f"import {module_name}"
        ref_src = f"{module_name}.{raw_type.__name__}"
    else:
        parts = raw_type.__module__.split(".")
        ref_src = f"{parts[-1]}.{raw_type.__name__}"
        if len(parts) > 1:
            import_src = f"from {'.'.join(parts[:-1])} import {parts[-1]}"
        else:
            import_src = f"import {parts[0]}"

    return _Source(src=ref_src, imports={import_src})


def _gen_nested_pydantic(raw_type: Any) -> _Source:
    """Handles `list[PydanticModel]` and similar, correctly resolving imports
    of model args that might be defined in other files."""
    origin = get_origin(raw_type)
    assert origin in framework._SIMPLE_CONTAINERS
    container = _gen_type_import_and_ref(private_types.TypeDescriptor(raw=origin))
    args = get_args(raw_type)
    arg_parts = []
    for arg in args:
        arg_src = _gen_type_import_and_ref(private_types.TypeDescriptor(raw=arg))
        arg_parts.append(arg_src.src)
        container.imports.update(arg_src.imports)

    container.src = f"{container.src}[{','.join(arg_parts)}]"
    return container


def _gen_type_import_and_ref(type_descr: private_types.TypeDescriptor) -> _Source:
    """Returns e.g. ("from sub_package import module", "module.OutputType")."""
    if type_descr.is_pydantic:
        return _gen_pydantic_import_and_ref(type_descr.raw)
    if type_descr.has_pydantic_args:
        return _gen_nested_pydantic(type_descr.raw)
    if isinstance(type_descr.raw, type):
        if not type_descr.raw.__module__ == "builtins":
            raise TypeError(
                f"{type_descr.raw} is not a builtin - cannot be rendered as source."
            )
        return _Source(src=type_descr.raw.__name__)

    return _Source(src=str(type_descr.raw))


def _gen_streaming_type_import_and_ref(
    stream_type: private_types.StreamingTypeDescriptor,
) -> _Source:
    """Unlike other `_gen`-helpers, this does not define a type, it creates a symbol."""
    mod = stream_type.origin_type.__module__
    arg = stream_type.arg_type.__name__
    type_src = f"{mod}.{stream_type.origin_type.__name__}[{arg}]"
    return _Source(src=type_src, imports={f"import {mod}"})


def _gen_chainlet_import_and_ref(
    chainlet_descriptor: private_types.ChainletAPIDescriptor,
) -> _Source:
    """Returns e.g. ("from sub_package import module", "module.OutputType")."""
    return _gen_pydantic_import_and_ref(chainlet_descriptor.chainlet_cls)


# I/O used by Stubs and Truss models ###################################################


def _get_input_model_name(chainlet_name: str) -> str:
    return f"{chainlet_name}Input"


def _get_output_model_name(chainlet_name: str) -> str:
    return f"{chainlet_name}Output"


def _gen_truss_input_pydantic(
    chainlet_descriptor: private_types.ChainletAPIDescriptor,
) -> _Source:
    imports = {"import pydantic", "from typing import Optional"}
    fields = []
    for arg in chainlet_descriptor.endpoint.input_args:
        type_ref = _gen_type_import_and_ref(arg.type)
        imports.update(type_ref.imports)
        if arg.is_optional:
            fields.append(f"{arg.name}: Optional[{type_ref.src}] = None")
        else:
            fields.append(f"{arg.name}: {type_ref.src}")

    if fields:
        field_block = _indent("\n".join(fields))
    else:
        field_block = _indent("pass")

    model_name = _get_input_model_name(chainlet_descriptor.name)
    src = f"class {model_name}(pydantic.BaseModel):\n{field_block}"
    return _Source(src=src, imports=imports)


def _gen_truss_output_pydantic(
    chainlet_descriptor: private_types.ChainletAPIDescriptor,
) -> _Source:
    imports = {"import pydantic"}
    fields: list[str] = []
    for i, output_type in enumerate(chainlet_descriptor.endpoint.output_types):
        _update_src(_gen_type_import_and_ref(output_type), fields, imports)

    model_name = _get_output_model_name(chainlet_descriptor.name)
    if len(fields) > 1:
        root_type = f"tuple[{','.join(fields)}]"
    else:
        root_type = fields[0]
    src = f"{model_name} = pydantic.RootModel[{root_type}]"
    return _Source(src=src, imports=imports)


# Stub Gen #############################################################################


def _stub_endpoint_signature_src(
    endpoint: private_types.EndpointAPIDescriptor,
) -> _Source:
    """
    E.g.:
    ```
    async def run_remote(
        self, inputs: shared_chainlet.SplitTextInput, extra_arg: int
    ) -> tuple[shared_chainlet.SplitTextOutput, int]:
    ```
    """
    imports = set()
    args = ["self"]
    for arg in endpoint.input_args:
        arg_ref = _gen_type_import_and_ref(arg.type)
        imports.update(arg_ref.imports)
        args.append(f"{arg.name}: {arg_ref.src}")

    if endpoint.is_streaming:
        streaming_src = _gen_streaming_type_import_and_ref(endpoint.streaming_type)
        imports.update(streaming_src.imports)
        output = streaming_src.src
    else:
        outputs: list[str] = []
        for output_type in endpoint.output_types:
            _update_src(_gen_type_import_and_ref(output_type), outputs, imports)

        if len(outputs) == 1:
            output = outputs[0]
        else:
            output = f"tuple[{', '.join(outputs)}]"

    def_str = "async def" if endpoint.is_async else "def"
    return _Source(
        src=f"{def_str} {endpoint.name}({','.join(args)}) -> {output}:", imports=imports
    )


def _stub_endpoint_body_src(
    endpoint: private_types.EndpointAPIDescriptor, chainlet_name: str
) -> _Source:
    """Generates source code for calling the stub and wrapping the I/O types.

    E.g.:
    ```
    return await self.predict_async(
        SplitTextInput(inputs=inputs, extra_arg=extra_arg), SplitTextOutput).root
    ```
    """
    imports: set[str] = set()
    if endpoint.has_engine_builder_llm_input:
        assert len(endpoint.input_args) == 1
        arg = endpoint.input_args[0]
        assert arg.name == "llm_input"
        # Since the deployed model is not a chainlet with generated top-level pydantic
        # input type, we pass values directly.
        inputs = "inputs=llm_input"
    else:
        args = [f"{arg.name}={arg.name}" for arg in endpoint.input_args]
        if args:
            inputs = f"{_get_input_model_name(chainlet_name)}({', '.join(args)})"
        else:
            inputs = "{}"

    parts = []
    # Invoke remote.
    if not endpoint.is_streaming:
        output_model_name = _get_output_model_name(chainlet_name)
        if endpoint.is_async:
            parts = [
                f"return (await self.predict_async({inputs}, {output_model_name})).root"
            ]
        else:
            parts = [f"return self.predict_sync({inputs}, {output_model_name}).root"]

    else:
        if endpoint.is_async:
            parts.append(
                f"async for data in await self.predict_async_stream({inputs}):"
            )
            if endpoint.streaming_type.is_string:
                parts.append(_indent("yield data.decode()"))
            else:
                parts.append(_indent("yield data"))
        else:
            raise NotImplementedError(
                "`Streaming endpoints (containing `yield` statements) are only "
                "supported for async endpoints."
            )

    return _Source(src="\n".join(parts), imports=imports)


def _gen_stub_src(chainlet: private_types.ChainletAPIDescriptor) -> _Source:
    """Generates stub class source, e.g:

    ```
    <IMPORTS>

    class SplitTextInput(pydantic.BaseModel):
        inputs: shared_chainlet.SplitTextInput
        extra_arg: int

    class SplitTextOutput(pydantic.BaseModel):
        output: tuple[shared_chainlet.SplitTextOutput, int]

    class SplitText(stub.StubBase):
        async def run_remote(
            self, inputs: shared_chainlet.SplitTextInput, extra_arg: int
        ) -> tuple[shared_chainlet.SplitTextOutput, int]:
            return await self.predict_async(
                SplitTextInput(inputs=inputs, extra_arg=extra_arg), SplitTextOutput).root
    ```
    """
    imports = {"from truss_chains.remote_chainlet import stub"}
    src_parts: list[str] = []
    if not framework.is_engine_builder_chainlet(chainlet.chainlet_cls):
        input_src = _gen_truss_input_pydantic(chainlet)
        _update_src(input_src, src_parts, imports)

    if not chainlet.endpoint.is_streaming:
        output_src = _gen_truss_output_pydantic(chainlet)
        _update_src(output_src, src_parts, imports)

    signature = _stub_endpoint_signature_src(chainlet.endpoint)
    imports.update(signature.imports)
    body = _stub_endpoint_body_src(chainlet.endpoint, chainlet.name)
    imports.update(body.imports)

    src_parts.extend(
        [
            f"class {chainlet.name}(stub.StubBase):",
            _indent(signature.src),
            _indent(body.src, 2),
            "\n",
        ]
    )
    return _Source(src="\n".join(src_parts), imports=imports)


def _gen_stub_src_for_deps(
    dependencies: Iterable[private_types.ChainletAPIDescriptor],
) -> Optional[_Source]:
    """Generates a source code and imports for stub classes."""
    imports: set[str] = set()
    src_parts: list[str] = []
    for dep in dependencies:
        _update_src(_gen_stub_src(dep), src_parts, imports)

    if not (imports or src_parts):
        return None
    return _Source(src="\n".join(src_parts), imports=imports)


# Truss Chainlet Gen ###################################################################


def _name_to_dirname(name: str) -> str:
    """Make a string safe to use as a directory name."""
    name = name.strip()  # Remove leading and trailing spaces
    name = re.sub(
        r"[^\w.-]", "_", name
    )  # Replace non-alphanumeric characters with underscores
    name = re.sub(r"_+", "_", name)  # Collapse multiple underscores into a single one
    return name


def _make_chainlet_dir(
    chain_name: str,
    chainlet_descriptor: private_types.ChainletAPIDescriptor,
    root: pathlib.Path,
) -> pathlib.Path:
    dir_name = f"chainlet_{chainlet_descriptor.name}"
    chainlet_dir = (
        root
        / private_types.GENERATED_CODE_DIR
        / _name_to_dirname(chain_name)
        / dir_name
    )
    if chainlet_dir.exists():
        shutil.rmtree(chainlet_dir)
    chainlet_dir.mkdir(exist_ok=False, parents=True)
    return chainlet_dir


class _SpecifyChainletTypeAnnotation(libcst.CSTTransformer):
    """Inserts the concrete chainlet class into `_chainlet: private_types.ABCChainlet`."""

    def __init__(self, new_annotation: str) -> None:
        super().__init__()
        self._new_annotation = libcst.parse_expression(new_annotation)

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
                and statement.target.value == "_chainlet"
            ):
                new_annotation = libcst.Annotation(annotation=self._new_annotation)
                new_statement = statement.with_changes(annotation=new_annotation)
                new_body.append(new_statement)
            else:
                new_body.append(statement)

        return updated_node.with_changes(body=tuple(new_body))


def _gen_load_src(chainlet_descriptor: private_types.ChainletAPIDescriptor) -> _Source:
    imports = {"from truss_chains.remote_chainlet import stub", "import logging"}
    stub_args = []
    for name, dep in chainlet_descriptor.dependencies.items():
        # `dep.name` is the class name, while `name` is the argument name.
        stub_args.append(f"{name}=stub.factory({dep.name}, self._context)")

    if chainlet_descriptor.has_context:
        if stub_args:
            init_args = f"{', '.join(stub_args)}, context=self._context"
        else:
            init_args = "context=self._context"
    else:
        init_args = ", ".join(stub_args)

    user_chainlet_ref = _gen_chainlet_import_and_ref(chainlet_descriptor)
    imports.update(user_chainlet_ref.imports)
    body = _indent(
        "\n".join(
            [f"logging.info(f'Loading Chainlet `{chainlet_descriptor.name}`.')"]
            + [f"self._chainlet = {user_chainlet_ref.src}({init_args})"]
        )
    )
    src = "\n".join(["def load(self) -> None:", body])
    return _Source(src=src, imports=imports)


def _gen_health_check_src(
    health_check: private_types.HealthCheckAPIDescriptor,
) -> _Source:
    def_str = "async def" if health_check.is_async else "def"
    maybe_await = "await " if health_check.is_async else ""
    src = (
        f"{def_str} is_healthy(self) -> Optional[bool]:\n"
        f"""{_indent('if hasattr(self, "_chainlet"):')}"""
        f"""{_indent(f"return {maybe_await}self._chainlet.is_healthy()")}"""
    )
    return _Source(src=src)


def _gen_predict_src(
    chainlet_descriptor: private_types.ChainletAPIDescriptor,
) -> _Source:
    imports: set[str] = {
        "from truss_chains.remote_chainlet import stub",
        "from truss_chains.remote_chainlet import utils",
    }
    parts: list[str] = []
    def_str = "async def" if chainlet_descriptor.endpoint.is_async else "def"
    input_model_name = _get_input_model_name(chainlet_descriptor.name)
    if chainlet_descriptor.endpoint.is_streaming:
        streaming_src = _gen_streaming_type_import_and_ref(
            chainlet_descriptor.endpoint.streaming_type
        )
        imports.update(streaming_src.imports)
        output_type_name = streaming_src.src
    else:
        output_type_name = _get_output_model_name(chainlet_descriptor.name)

    imports.add("import starlette.requests")
    parts.append(
        f"{def_str} predict(self, inputs: {input_model_name}, "
        f"request: starlette.requests.Request) -> {output_type_name}:"
    )
    # Add error handling context manager:
    parts.append(_indent("with utils.predict_context(request.headers):"))
    # Invoke Chainlet.
    if (
        chainlet_descriptor.endpoint.is_async
        and not chainlet_descriptor.endpoint.is_streaming
    ):
        maybe_await = "await "
    else:
        maybe_await = ""
    run_remote = chainlet_descriptor.endpoint.name
    # See docs of `pydantic_set_field_dict` for why this is needed.
    args = "**utils.pydantic_set_field_dict(inputs)"
    parts.append(
        _indent(f"result = {maybe_await}self._chainlet.{run_remote}({args})", 2)
    )
    if chainlet_descriptor.endpoint.is_streaming:
        # Streaming returns raw iterator, no pydantic model.
        # This needs to be nested inside the `trace_parent` context!
        parts.append(_indent("async for chunk in result:", 2))
        parts.append(_indent("yield chunk", 3))
    else:
        result_pydantic = f"{output_type_name}(result)"
        parts.append(_indent(f"return {result_pydantic}"))
    return _Source(src="\n".join(parts), imports=imports)


def _gen_websocket_src() -> _Source:
    src = """
async def websocket(self, websocket: fastapi.WebSocket) -> None:
    with utils.predict_context(websocket.headers):
        await self._chainlet.run_remote(
            utils.WebsocketWrapperFastAPI(websocket)
        )"""
    return _Source(
        src=src,
        imports={"import fastapi", "from truss_chains.remote_chainlet import utils"},
    )


def _gen_truss_chainlet_model(
    chainlet_descriptor: private_types.ChainletAPIDescriptor,
) -> _Source:
    skeleton_tree = libcst.parse_module(_MODEL_SKELETON_FILE.read_text())
    imports: set[str] = set(
        libcst.Module(body=[node]).code
        for node in skeleton_tree.body
        if isinstance(node, libcst.SimpleStatementLine)
        and any(
            isinstance(stmt, libcst.Import) or isinstance(stmt, libcst.ImportFrom)
            for stmt in node.body
        )
    )
    class_definition: libcst.ClassDef = utils.expect_one(
        node
        for node in skeleton_tree.body
        if isinstance(node, libcst.ClassDef) and node.name.value == _MODEL_CLS_NAME
    )

    load_src = _gen_load_src(chainlet_descriptor)
    imports.update(load_src.imports)
    if chainlet_descriptor.endpoint.is_websocket:
        endpoint_src = _gen_websocket_src()
    else:
        endpoint_src = _gen_predict_src(chainlet_descriptor)

    imports.update(endpoint_src.imports)
    new_body: list[Any] = list(class_definition.body.body) + [
        libcst.parse_statement(load_src.src),
        libcst.parse_statement(endpoint_src.src),
    ]

    if chainlet_descriptor.health_check is not None:
        health_check_src = _gen_health_check_src(chainlet_descriptor.health_check)
        new_body.extend([libcst.parse_statement(health_check_src.src)])

    user_chainlet_ref = _gen_chainlet_import_and_ref(chainlet_descriptor)
    imports.update(user_chainlet_ref.imports)

    new_block = libcst.IndentedBlock(body=new_body)
    class_definition = class_definition.with_changes(body=new_block)
    class_definition = class_definition.visit(  # type: ignore[assignment]
        _SpecifyChainletTypeAnnotation(user_chainlet_ref.src)
    )
    model_class_src = libcst.Module(body=[class_definition]).code
    return _Source(src=model_class_src, imports=imports)


def _gen_truss_chainlet_file(
    chainlet_dir: pathlib.Path,
    chainlet_descriptor: private_types.ChainletAPIDescriptor,
    dependencies: Iterable[private_types.ChainletAPIDescriptor],
) -> pathlib.Path:
    """Generates code that wraps a Chainlet as a truss-compatible model."""
    file_path = chainlet_dir / truss_config.DEFAULT_MODEL_MODULE_DIR / _MODEL_FILENAME
    file_path.parent.mkdir(parents=True, exist_ok=True)
    (chainlet_dir / truss_config.DEFAULT_MODEL_MODULE_DIR / "__init__.py").touch()
    imports: set[str] = set()
    src_parts: list[str] = []

    if maybe_stub_src := _gen_stub_src_for_deps(dependencies):
        _update_src(maybe_stub_src, src_parts, imports)

    if chainlet_descriptor.endpoint.has_pydantic_input:
        input_src = _gen_truss_input_pydantic(chainlet_descriptor)
        _update_src(input_src, src_parts, imports)

    if chainlet_descriptor.endpoint.has_pydantic_output:
        output_src = _gen_truss_output_pydantic(chainlet_descriptor)
        _update_src(output_src, src_parts, imports)

    model_src = _gen_truss_chainlet_model(chainlet_descriptor)
    _update_src(model_src, src_parts, imports)

    imports_str = "\n".join(imports)
    src_str = "\n".join(src_parts)
    file_path.write_text(f"{imports_str}\n{src_str}")
    _format_python_file(file_path)
    return file_path


# Truss Gen ############################################################################


def _make_requirements(image: public_types.DockerImage) -> list[str]:
    """Merges file- and list-based requirements and adds truss git if not present."""
    pip_requirements: set[str] = set()
    if image.pip_requirements_file:
        pip_requirements.update(
            req
            for req in pathlib.Path(image.pip_requirements_file.abs_path)
            .read_text()
            .splitlines()
            if not req.strip().startswith("#")
        )
    pip_requirements.update(image.pip_requirements)

    truss_pypy = next(
        (req for req in pip_requirements if _TRUSS_PIP_PATTERN.match(req)), None
    )

    truss_git = next((req for req in pip_requirements if _TRUSS_GIT in req), None)

    if truss_git:
        logging.warning(
            "The chainlet contains a truss version from github as a pip_requirement:\n"
            f"\t{truss_git}\n"
            "This could result in inconsistencies between the deploying client and the "
            "deployed chainlet. This is not recommended for production chains."
        )
    if truss_pypy:
        logging.warning(
            "The chainlet contains a pinned truss version as a pip_requirement:\n"
            f"\t{truss_pypy}\n"
            "This could result in inconsistencies between the deploying client and the "
            "deployed chainlet. This is not recommended for production chains. If "
            "`truss` is not manually added as a requirement, the same version as "
            "locally installed will be automatically added and ensure compatibility."
        )

    if not (truss_git or truss_pypy):
        truss_pip = f"truss=={truss.version()}"
        logging.debug(
            f"Truss not found in pip requirements, auto-adding: `{truss_pip}`."
        )
        pip_requirements.add(truss_pip)

    return sorted(pip_requirements)


def _inplace_fill_base_image(
    image: public_types.DockerImage, mutable_truss_config: truss_config.TrussConfig
) -> None:
    if isinstance(image.base_image, public_types.BasetenImage):
        mutable_truss_config.python_version = image.base_image.value
    elif isinstance(image.base_image, public_types.CustomImage):
        mutable_truss_config.base_image = truss_config.BaseImage(
            image=image.base_image.image, docker_auth=image.base_image.docker_auth
        )
        if image.base_image.python_executable_path:
            mutable_truss_config.base_image.python_executable_path = (
                image.base_image.python_executable_path
            )
    elif isinstance(image.base_image, str):  # This options is deprecated.
        raise NotImplementedError(
            "Specifying docker base image as string is deprecated"
        )


def _gen_truss_config(
    chainlet_dir: pathlib.Path,
    chainlet_descriptor: private_types.ChainletAPIDescriptor,
    chainlet_to_service: Mapping[str, private_types.ServiceDescriptor],
    model_name: str,
    use_local_src: bool,
) -> truss_config.TrussConfig:
    """Generate a truss config for a Chainlet."""
    config = truss_config.TrussConfig()
    config.model_name = model_name
    remote_config = chainlet_descriptor.chainlet_cls.remote_config

    # Compute.
    compute = remote_config.get_compute_spec()
    config.resources.cpu = str(compute.cpu_count)
    config.resources.memory = str(compute.memory)
    config.resources.accelerator = compute.accelerator
    config.runtime.predict_concurrency = compute.predict_concurrency
    config.runtime.is_websocket_endpoint = chainlet_descriptor.endpoint.is_websocket

    assets = remote_config.get_asset_spec()
    config.secrets = {k: v for k, v in assets.secrets.items()}
    config.runtime.enable_tracing_data = remote_config.options.enable_b10_tracing
    config.runtime.enable_debug_logs = remote_config.options.enable_debug_logs
    config.model_metadata = cast(dict[str, Any], remote_config.options.metadata) or {}
    config.environment_variables = dict(remote_config.options.env_variables)

    if issubclass(chainlet_descriptor.chainlet_cls, framework.EngineBuilderChainlet):
        config.trt_llm = chainlet_descriptor.chainlet_cls.engine_builder_config
        truss_config.TrussConfig.model_validate(config)
        return config

    config.model_class_filename = _MODEL_FILENAME
    config.model_class_name = _MODEL_CLS_NAME

    config.runtime.health_checks = remote_config.options.health_checks
    # Image.
    _inplace_fill_base_image(remote_config.docker_image, config)
    pip_requirements = _make_requirements(remote_config.docker_image)
    # TODO: `pip_requirements` will add server requirements which give version
    #  conflicts. Check if that's still the case after relaxing versions.
    # config.requirements = pip_requirements
    pip_requirements_file_path = chainlet_dir / _REQUIREMENTS_FILENAME
    pip_requirements_file_path.write_text("\n".join(pip_requirements))
    # Absolute paths don't work with remote build.
    config.requirements_file = _REQUIREMENTS_FILENAME
    config.system_packages = remote_config.docker_image.apt_requirements
    if remote_config.docker_image.external_package_dirs:
        for ext_dir in remote_config.docker_image.external_package_dirs:
            config.external_package_dirs.append(ext_dir.abs_path)
    config.use_local_src = use_local_src

    if public_types._BASETEN_API_SECRET_NAME not in config.secrets:
        config.secrets[public_types._BASETEN_API_SECRET_NAME] = (
            public_types.SECRET_DUMMY
        )
    else:
        logging.info(
            f"Chains automatically add {public_types._BASETEN_API_SECRET_NAME} "
            "to secrets - no need to manually add it."
        )
    config.model_cache = truss_config.ModelCache(assets.cached)
    config.external_data = truss_config.ExternalData(assets.external_data)
    config.model_metadata[private_types.TRUSS_CONFIG_CHAINS_KEY] = (
        private_types.TrussMetadata(
            chainlet_to_service=chainlet_to_service
        ).model_dump()
    )
    truss_config.TrussConfig.model_validate(config)
    return config


def gen_truss_model_from_source(
    model_src: pathlib.Path, use_local_src: bool = False
) -> pathlib.Path:
    # TODO(nikhil): Improve detection of directory structure, since right now
    # we assume a flat structure
    root_dir = model_src.absolute().parent
    with framework.ModelImporter.import_target(model_src) as entrypoint_cls:
        descriptor = framework.get_descriptor(entrypoint_cls)
        return gen_truss_model(
            model_root=root_dir,
            model_name=entrypoint_cls.display_name,
            model_descriptor=descriptor,
            use_local_src=use_local_src,
        )


def gen_truss_model(
    model_root: pathlib.Path,
    model_name: str,
    model_descriptor: private_types.ChainletAPIDescriptor,
    use_local_src: bool = False,
) -> pathlib.Path:
    return gen_truss_chainlet(
        chain_root=model_root,
        chain_name=model_name,
        chainlet_descriptor=model_descriptor,
        use_local_src=use_local_src,
    )


def gen_truss_chainlet(
    chain_root: pathlib.Path,
    chain_name: str,
    chainlet_descriptor: private_types.ChainletAPIDescriptor,
    model_name: Optional[str] = None,
    use_local_src: bool = False,
) -> pathlib.Path:
    # Filter needed services and customize options.
    dep_services = {}
    for dep in chainlet_descriptor.dependencies.values():
        dep_services[dep.name] = private_types.ServiceDescriptor(
            name=dep.name, display_name=dep.display_name, options=dep.options
        )
    gen_root = pathlib.Path(tempfile.gettempdir())
    chainlet_dir = _make_chainlet_dir(chain_name, chainlet_descriptor, gen_root)
    logging.info(
        f"Code generation for {chainlet_descriptor.chainlet_cls.entity_type} `{chainlet_descriptor.name}` "
        f"in `{chainlet_dir}`."
    )
    if framework.is_engine_builder_chainlet(chainlet_descriptor.chainlet_cls):
        engine_builder_config = cast(
            framework.EngineBuilderChainlet, chainlet_descriptor.chainlet_cls
        ).engine_builder_config
    else:
        engine_builder_config = None

    config = _gen_truss_config(
        chainlet_dir,
        chainlet_descriptor,
        dep_services,
        model_name=model_name or chain_name,
        use_local_src=use_local_src,
    )
    config.write_to_yaml_file(
        chainlet_dir / serving_image_builder.CONFIG_FILE, verbose=True
    )
    if engine_builder_config:
        return chainlet_dir

    # This assumes all imports are absolute w.r.t chain root (or site-packages).
    truss_path.copy_tree_path(
        chain_root, chainlet_dir / truss_config.DEFAULT_BUNDLED_PACKAGES_DIR
    )
    for file in chain_root.glob("*.py"):
        if "-" in file.name:
            raise public_types.ChainsUsageError(
                f"Python file `{file}` contains `-`, use `_` instead."
            )
        if file.name == _MODEL_FILENAME:
            raise public_types.ChainsUsageError(
                f"Python file name `{_MODEL_FILENAME}` is reserved and cannot be used."
            )
    chainlet_file = _gen_truss_chainlet_file(
        chainlet_dir,
        chainlet_descriptor,
        framework.get_dependencies(chainlet_descriptor),
    )
    remote_config = chainlet_descriptor.chainlet_cls.remote_config
    if remote_config.docker_image.data_dir:
        data_dir = chainlet_dir / truss_config.DEFAULT_DATA_DIRECTORY
        data_dir.mkdir(parents=True, exist_ok=True)
        user_data_dir = remote_config.docker_image.data_dir.abs_path
        shutil.copytree(user_data_dir, data_dir, dirs_exist_ok=True)

    # Copy model file s.t. during debugging imports can are properly resolved.
    shutil.copy(
        chainlet_file,
        chainlet_file.parent.parent
        / truss_config.DEFAULT_BUNDLED_PACKAGES_DIR
        / "_model_dbg.py",
    )
    return chainlet_dir

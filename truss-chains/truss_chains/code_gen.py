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
import textwrap
from typing import Any, Iterable, Mapping, Optional

import libcst
import truss
from truss import truss_config
from truss.contexts.image_builder import serving_image_builder
from truss.util import path as truss_path

from truss_chains import definitions, framework, model_skeleton, utils

INDENT = " " * 4
_REQUIREMENTS_FILENAME = "pip_requirements.txt"
_MODEL_FILENAME = "model.py"
_MODEL_CLS_NAME = model_skeleton.TrussChainletModel.__name__
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


def _indent(text: str, num: int = 1) -> str:
    return textwrap.indent(text, INDENT * num)


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


class _Source(definitions.SafeModelNonSerializable):
    src: str
    imports: set[str] = set()


def _update_src(new_source: _Source, src_parts: list[str], imports: set[str]) -> None:
    src_parts.append(new_source.src)
    imports.update(new_source.imports)


def _gen_import_and_ref(raw_type: Any) -> _Source:
    """Returns e.g. ("from sub_package import module", "module.OutputType")."""
    if raw_type.__module__ == "__main__":
        # TODO: assuming that main is copied into package dir and can be imported.
        module_obj = sys.modules[raw_type.__module__]
        if not module_obj.__file__:
            raise definitions.ChainsUsageError(
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


def _gen_type_import_and_ref(type_descr: definitions.TypeDescriptor) -> _Source:
    """Returns e.g. ("from sub_package import module", "module.OutputType")."""
    if type_descr.is_pydantic:
        return _gen_import_and_ref(type_descr.raw)

    elif isinstance(type_descr.raw, type):
        if not type_descr.raw.__module__ == "builtins":
            raise TypeError(
                f"{type_descr.raw} is not a builtin - cannot be rendered as source."
            )
        return _Source(src=type_descr.raw.__name__)
    else:
        return _Source(src=str(type_descr.raw))


def _gen_chainlet_import_and_ref(
    chainlet_descriptor: definitions.ChainletAPIDescriptor,
) -> _Source:
    """Returns e.g. ("from sub_package import module", "module.OutputType")."""
    return _gen_import_and_ref(chainlet_descriptor.chainlet_cls)


# I/O used by Stubs and Truss models ###################################################


def _get_input_model_name(chainlet_name: str) -> str:
    return f"{chainlet_name}Input"


def _get_output_model_name(chainlet_name: str) -> str:
    return f"{chainlet_name}Output"


def _gen_truss_input_pydantic(
    chainlet_descriptor: definitions.ChainletAPIDescriptor,
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
    chainlet_descriptor: definitions.ChainletAPIDescriptor,
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
    endpoint: definitions.EndpointAPIDescriptor,
) -> _Source:
    """
    E.g.:
    ```
    async def run_remote(
        self, inputs: shared_chainlet.SplitTextInput, extra_arg: int
    ) -> tuple[shared_chainlet.SplitTextOutput, int]:
    ```
    """
    if endpoint.is_generator:
        raise NotImplementedError("Generator.")

    imports = set()
    args = []
    for arg in endpoint.input_args:
        arg_ref = _gen_type_import_and_ref(arg.type)
        imports.update(arg_ref.imports)
        args.append(f"{arg.name}: {arg_ref.src}")

    outputs: list[str] = []
    for output_type in endpoint.output_types:
        _update_src(_gen_type_import_and_ref(output_type), outputs, imports)

    if len(outputs) == 1:
        output = outputs[0]
    else:
        output = f"tuple[{', '.join(outputs)}]"

    def_str = "async def" if endpoint.is_async else "def"
    return _Source(
        src=f"{def_str} {endpoint.name}(self, {','.join(args)}) -> {output}:",
        imports=imports,
    )


def _stub_endpoint_body_src(
    endpoint: definitions.EndpointAPIDescriptor, chainlet_name: str
) -> _Source:
    """Generates source code for calling the stub and wrapping the I/O types.

    E.g.:
    ```
    json_result = await self._remote.predict_async(
        SplitTextInput(inputs=inputs, extra_arg=extra_arg).model_dump())
    return SplitTextOutput.model_validate(json_result).output
    ```
    """
    if endpoint.is_generator:
        raise NotImplementedError("Generator")

    imports: set[str] = set()
    args = [f"{arg.name}={arg.name}" for arg in endpoint.input_args]
    inputs = f"{_get_input_model_name(chainlet_name)}({', '.join(args)}).model_dump()"

    # Invoke remote.
    if endpoint.is_async:
        remote_call = f"await self._remote.predict_async({inputs})"
    else:
        remote_call = f"self._remote.predict_sync({inputs})"

    parts = [f"json_result = {remote_call}"]
    # Unpack response and parse as pydantic models if needed.
    output_model_name = _get_output_model_name(chainlet_name)
    parts.append(f"return {output_model_name}.model_validate(json_result).root")
    return _Source(src="\n".join(parts), imports=imports)


def _gen_stub_src(chainlet: definitions.ChainletAPIDescriptor) -> _Source:
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
            json_result = await self._remote.predict_async(
                SplitTextInput(inputs=inputs, extra_arg=extra_arg).model_dump())
            return SplitTextOutput.model_validate(json_result).root
    ```
    """
    imports = {"from truss_chains import stub"}
    src_parts: list[str] = []
    input_src = _gen_truss_input_pydantic(chainlet)
    _update_src(input_src, src_parts, imports)
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
    dependencies: Iterable[definitions.ChainletAPIDescriptor],
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


def _make_chainlet_dir(
    chain_name: str,
    chainlet_descriptor: definitions.ChainletAPIDescriptor,
    root: pathlib.Path,
) -> pathlib.Path:
    dir_name = f"chainlet_{chainlet_descriptor.name}"
    chainlet_dir = root / definitions.GENERATED_CODE_DIR / chain_name / dir_name
    if chainlet_dir.exists():
        shutil.rmtree(chainlet_dir)
    chainlet_dir.mkdir(exist_ok=False, parents=True)
    return chainlet_dir


class _SpecifyChainletTypeAnnotation(libcst.CSTTransformer):
    """Inserts the concrete chainlet class into `_chainlet: definitions.ABCChainlet`."""

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


def _gen_load_src(chainlet_descriptor: definitions.ChainletAPIDescriptor) -> _Source:
    """Generates AST for the `load` method of the truss model."""
    imports = {"from truss_chains import stub", "import logging"}
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


def _gen_predict_src(chainlet_descriptor: definitions.ChainletAPIDescriptor) -> _Source:
    """Generates AST for the `predict` method of the truss model."""
    if chainlet_descriptor.endpoint.is_generator:
        raise NotImplementedError("Generator.")

    imports: set[str] = {"from truss_chains import utils"}
    parts: list[str] = []
    def_str = "async def" if chainlet_descriptor.endpoint.is_async else "def"
    input_model_name = _get_input_model_name(chainlet_descriptor.name)
    output_model_name = _get_output_model_name(chainlet_descriptor.name)
    parts.append(
        f"{def_str} predict(self, inputs: {input_model_name}) "
        f"-> {output_model_name}:"
    )
    # Add error handling context manager:
    parts.append(
        _indent(
            f"with utils.exception_to_http_error("
            f'include_stack=True, chainlet_name="{chainlet_descriptor.name}"):'
        )
    )
    # Invoke Chainlet.
    maybe_await = "await " if chainlet_descriptor.endpoint.is_async else ""
    run_remote = chainlet_descriptor.endpoint.name
    # `exclude_unset` is important to handle arguments where `run_remote` has a default
    # correctly. In that case the pydantic model has an optional field and defaults to
    # `None`. But there might also be situations where the user explicitly passes a
    # value of `None`. So the condition whether to pass that argument or not is
    # whether it was *set* in the model. It is considered unset, if the incoming JSON
    # (from which the model was parsed/initialized) does not have that key.
    args = "**utils.pydantic_set_field_dict(inputs)"
    parts.append(
        _indent(f"result = {maybe_await}self._chainlet.{run_remote}({args})", 2)
    )
    result_pydantic = f"{output_model_name}(result)"
    parts.append(_indent(f"return {result_pydantic}"))
    return _Source(src="\n".join(parts), imports=imports)


def _gen_truss_chainlet_model(
    chainlet_descriptor: definitions.ChainletAPIDescriptor,
) -> _Source:
    skeleton_tree = libcst.parse_module(
        pathlib.Path(model_skeleton.__file__).read_text()
    )
    imports: set[str] = set(
        libcst.Module(body=[node]).code
        for node in skeleton_tree.body
        if isinstance(node, libcst.SimpleStatementLine)
        and any(
            isinstance(stmt, libcst.Import) or isinstance(stmt, libcst.ImportFrom)
            for stmt in node.body
        )
    )

    imports.add("import logging")
    imports.add("from truss_chains import utils")

    class_definition: libcst.ClassDef = utils.expect_one(
        node
        for node in skeleton_tree.body
        if isinstance(node, libcst.ClassDef)
        and node.name.value == model_skeleton.TrussChainletModel.__name__
    )

    load_src = _gen_load_src(chainlet_descriptor)
    imports.update(load_src.imports)
    predict_src = _gen_predict_src(chainlet_descriptor)
    imports.update(predict_src.imports)

    new_body: list[Any] = list(class_definition.body.body) + [
        libcst.parse_statement(load_src.src),
        libcst.parse_statement(predict_src.src),
    ]

    user_chainlet_ref = _gen_chainlet_import_and_ref(chainlet_descriptor)
    imports.update(user_chainlet_ref.imports)

    new_block = libcst.IndentedBlock(body=new_body)
    class_definition = class_definition.with_changes(body=new_block)
    class_definition = class_definition.visit(  # type: ignore[assignment]
        _SpecifyChainletTypeAnnotation(user_chainlet_ref.src)
    )
    model_class_src = libcst.Module(body=[class_definition]).code

    if utils.issubclass_safe(chainlet_descriptor.user_config_type.raw, type(None)):
        userconfig_pin = "UserConfigT = type(None)"
    else:
        user_config_ref = _gen_type_import_and_ref(chainlet_descriptor.user_config_type)
        imports.update(user_config_ref.imports)
        userconfig_pin = f"UserConfigT = {user_config_ref.src}"
    return _Source(src=f"{userconfig_pin}\n\n{model_class_src}", imports=imports)


def _gen_truss_chainlet_file(
    chainlet_dir: pathlib.Path,
    chainlet_descriptor: definitions.ChainletAPIDescriptor,
    dependencies: Iterable[definitions.ChainletAPIDescriptor],
) -> pathlib.Path:
    """Generates code that wraps a Chainlet as a truss-compatible model."""
    file_path = chainlet_dir / truss_config.DEFAULT_MODEL_MODULE_DIR / _MODEL_FILENAME
    file_path.parent.mkdir(parents=True, exist_ok=True)
    (chainlet_dir / truss_config.DEFAULT_MODEL_MODULE_DIR / "__init__.py").touch()
    imports: set[str] = set()
    src_parts: list[str] = []
    if maybe_stub_src := _gen_stub_src_for_deps(dependencies):
        _update_src(maybe_stub_src, src_parts, imports)

    input_src = _gen_truss_input_pydantic(chainlet_descriptor)
    _update_src(input_src, src_parts, imports)
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


def _make_requirements(image: definitions.DockerImage) -> list[str]:
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

    has_truss_pypy = any(
        bool(_TRUSS_PIP_PATTERN.match(req)) for req in pip_requirements
    )
    has_truss_git = any(_TRUSS_GIT in req for req in pip_requirements)

    if not (has_truss_git or has_truss_pypy):
        truss_pip = f"truss=={truss.version()}"
        logging.info(
            f"Truss not found in pip requirements, auto-adding: `{truss_pip}`."
        )
        pip_requirements.add(truss_pip)

    return sorted(pip_requirements)


def _inplace_fill_base_image(
    image: definitions.DockerImage, mutable_truss_config: truss_config.TrussConfig
) -> None:
    if isinstance(image.base_image, definitions.BasetenImage):
        mutable_truss_config.python_version = image.base_image.value
    elif isinstance(image.base_image, definitions.CustomImage):
        mutable_truss_config.base_image = truss_config.BaseImage(
            image=image.base_image.image, docker_auth=image.base_image.docker_auth
        )
        if image.base_image.python_executable_path:
            mutable_truss_config.base_image.python_executable_path = (
                image.base_image.python_executable_path
            )

    elif isinstance(image.base_image, str):  # This options is deprecated.
        mutable_truss_config.base_image = truss_config.BaseImage(image=image.base_image)


def _make_truss_config(
    chainlet_dir: pathlib.Path,
    chains_config: definitions.RemoteConfig,
    user_config: definitions.UserConfigT,
    chainlet_to_service: Mapping[str, definitions.ServiceDescriptor],
    model_name: str,
    user_env: Mapping[str, str],
) -> truss_config.TrussConfig:
    """Generate a truss config for a Chainlet."""
    config = truss_config.TrussConfig()
    config.model_name = model_name
    config.model_class_filename = _MODEL_FILENAME
    config.model_class_name = _MODEL_CLS_NAME
    # Compute.
    compute = chains_config.get_compute_spec()
    config.resources.cpu = str(compute.cpu_count)
    config.resources.memory = str(compute.memory)
    config.resources.accelerator = compute.accelerator
    config.resources.use_gpu = bool(compute.accelerator.count)
    config.runtime.predict_concurrency = compute.predict_concurrency
    # Image.
    _inplace_fill_base_image(chains_config.docker_image, config)
    pip_requirements = _make_requirements(chains_config.docker_image)
    # TODO: `pip_requirements` will add server requirements which give version
    #  conflicts. Check if that's still the case after relaxing versions.
    # config.requirements = pip_requirements
    pip_requirements_file_path = chainlet_dir / _REQUIREMENTS_FILENAME
    pip_requirements_file_path.write_text("\n".join(pip_requirements))
    # Absolute paths don't work with remote build.
    config.requirements_file = _REQUIREMENTS_FILENAME
    config.system_packages = chains_config.docker_image.apt_requirements
    if chains_config.docker_image.external_package_dirs:
        for ext_dir in chains_config.docker_image.external_package_dirs:
            config.external_package_dirs.append(ext_dir.abs_path)
    # Assets.
    assets = chains_config.get_asset_spec()
    config.secrets = assets.secrets
    if definitions.BASETEN_API_SECRET_NAME not in config.secrets:
        config.secrets[definitions.BASETEN_API_SECRET_NAME] = definitions.SECRET_DUMMY
    else:
        logging.info(
            f"Chains automatically add {definitions.BASETEN_API_SECRET_NAME} "
            "to secrets - no need to manually add it."
        )
    config.model_cache.models = assets.cached
    config.external_data = truss_config.ExternalData(items=assets.external_data)
    # Metadata.
    chains_metadata: definitions.TrussMetadata = definitions.TrussMetadata(
        user_config=user_config,
        chainlet_to_service=chainlet_to_service,
        user_env=user_env,
    )
    config.model_metadata[definitions.TRUSS_CONFIG_CHAINS_KEY] = (
        chains_metadata.model_dump()
    )
    config.write_to_yaml_file(
        chainlet_dir / serving_image_builder.CONFIG_FILE, verbose=True
    )
    return config


def gen_truss_chainlet(
    chain_root: pathlib.Path,
    gen_root: pathlib.Path,
    chain_name: str,
    chainlet_descriptor: definitions.ChainletAPIDescriptor,
    model_name: str,
    chainlet_display_name_to_url: Mapping[str, str],
    user_env: Mapping[str, str],
) -> pathlib.Path:
    dependencies = framework.global_chainlet_registry.get_dependencies(
        chainlet_descriptor
    )
    # Filter needed services and customize options.
    dep_services = {}
    for dep in chainlet_descriptor.dependencies.values():
        dep_services[dep.name] = definitions.ServiceDescriptor(
            name=dep.name,
            predict_url=chainlet_display_name_to_url[dep.display_name],
            options=dep.options,
        )

    chainlet_dir = _make_chainlet_dir(chain_name, chainlet_descriptor, gen_root)

    _make_truss_config(
        chainlet_dir,
        chainlet_descriptor.chainlet_cls.remote_config,
        chainlet_descriptor.chainlet_cls.default_user_config,
        dep_services,
        model_name,
        user_env,
    )
    # TODO This assumes all imports are absolute w.r.t chain root (or site-packages).
    truss_path.copy_tree_path(
        chain_root, chainlet_dir / truss_config.DEFAULT_BUNDLED_PACKAGES_DIR
    )
    for file in chain_root.glob("*.py"):
        if "-" in file.name:
            raise definitions.ChainsUsageError(
                f"Python file `{file}` contains `-`, use `_` instead."
            )
        if file.name == _MODEL_FILENAME:
            raise definitions.ChainsUsageError(
                f"Python file name `{_MODEL_FILENAME}` is reserved and cannot be used."
            )
    chainlet_file = _gen_truss_chainlet_file(
        chainlet_dir, chainlet_descriptor, dependencies
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

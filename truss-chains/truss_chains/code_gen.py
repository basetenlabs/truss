"""
Chains currently assumes that everything from the directory in which the entrypoint
is defined (i.e. sibling files and nested dirs) could be imported/used. e.g.:

```
workspace/
  entrypoint.py
  helper.py
  some_package/
    utils.py
    sub_package/
      ...
```

These sources are copied into truss's `/packages` and can be imported on the remote.
Using code *outside* of the workspace is not supported:

```
shared_lib/
  common.py
workspace/
  entrypoint.py
  ...
```

`shared_lib` can only be imported on the remote if its installed as a pip
requirement (site-package), it will not be copied from the local host.
"""


import logging
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import textwrap
from typing import Any, Iterable, Mapping, Optional

import libcst
from truss import truss_config
from truss.contexts.image_builder import serving_image_builder
from truss_chains import definitions, model_skeleton, utils

_REQUIREMENTS_FILENAME = "pip_requirements.txt"
_MODEL_FILENAME = "model.py"
_MODEL_CLS_NAME = model_skeleton.TrussChainletModel.__name__


INDENT = " " * 4


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
    _run_simple_subprocess(
        f"autoflake --in-place --remove-all-unused-imports {file_path}"
    )
    _run_simple_subprocess(f"black {file_path}")
    _run_simple_subprocess(f"isort {file_path}")


class _Source(definitions.SafeModelNonSerializable):
    src: str
    imports: set[str] = set()


def _gen_import_and_ref(raw_type: Any) -> _Source:
    """Returns e.g. ("from user_package import module", "module.OutputType")."""
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
    """Returns e.g. ("from user_package import module", "module.OutputType")."""
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
    """Returns e.g. ("from user_package import module", "module.OutputType")."""
    return _gen_import_and_ref(chainlet_descriptor.chainlet_cls)


# Stub Gen #############################################################################


def _endpoint_signature_src(endpoint: definitions.EndpointAPIDescriptor) -> _Source:
    """
    E.g.: `async def run(self, data: str, num_partitions: int) -> tuple[list, int]:`
    """
    if endpoint.is_generator:
        # TODO: implement generator.
        raise NotImplementedError("Generator.")

    imports = set()
    args = []
    for arg_name, arg_type in endpoint.input_names_and_types:
        arg_ref = _gen_type_import_and_ref(arg_type)
        imports.update(arg_ref.imports)
        args.append(f"{arg_name}: {arg_ref.src}")

    outputs = []
    for output_type in endpoint.output_types:
        out_ref = _gen_type_import_and_ref(output_type)
        outputs.append(out_ref.src)
        imports.update(out_ref.imports)

    if len(outputs) == 1:
        output = outputs[0]
    else:
        output = f"tuple[{', '.join(outputs)}]"

    def_str = "async def" if endpoint.is_async else "def"
    return _Source(
        src=f"{def_str} {endpoint.name}(self, {','.join(args)}) -> {output}:",
        imports=imports,
    )


def _endpoint_body_src(endpoint: definitions.EndpointAPIDescriptor) -> _Source:
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

    imports = set()
    parts = []
    # Pack arg list as json dictionary.
    json_args = []
    for arg_name, arg_type in endpoint.input_names_and_types:
        if arg_type.is_pydantic:
            json_args.append(f"'{arg_name}': {arg_name}.dict()")
        else:
            json_args.append(f"'{arg_name}': {arg_name}")
    parts.append(f"json_args = {{{', '.join(json_args)}}}")

    # Invoke remote.
    if endpoint.is_async:
        remote_call = "await self._remote.predict_async(json_args)"
    else:
        remote_call = "self._remote.predict_sync(json_args)"

    parts.append(f"json_result = {remote_call}")

    # Unpack response and parse as pydantic models if needed.
    if len(endpoint.output_types) == 1:
        output_type = utils.expect_one(endpoint.output_types)
        if output_type.is_pydantic:
            out_ref = _gen_type_import_and_ref(output_type)
            imports.update(out_ref.imports)
            result = f"{out_ref.src}.parse_obj(json_result)"
        else:
            result = "json_result"
    else:
        outputs = []
        for i, output_type in enumerate(endpoint.output_types):
            if output_type.is_pydantic:
                out_ref = _gen_type_import_and_ref(output_type)
                outputs.append(f"{out_ref.src}.parse_obj(json_result[{i}])")
                imports.update(out_ref.imports)
            else:
                outputs.append(f"json_result[{i}]")

        result = ", ".join(outputs)

    parts.append(f"return {result}")
    return _Source(src="\n".join(parts), imports=imports)


def _gen_stub_src(chainlet: definitions.ChainletAPIDescriptor) -> _Source:
    """Generates stub class source, e.g:

    ```
    from truss_chains import stub

    class SplitText(stub.StubBase):
        def __init__(self, url: str, api_key: str) -> None:
            self._remote = stub.BasetenSession(url, api_key)

        async def run(self, data: str, num_partitions: int) -> tuple[SplitTextOutput, int]:
            json_args = {"inputs": inputs.dict(), "extra_arg": extra_arg}
            json_result = await self._remote.predict_async(json_args)
            return (SplitTextOutput.parse_obj(json_result[0]), json_result[1])
    ```
    """
    imports = {"from truss_chains import stub"}
    signature = _endpoint_signature_src(chainlet.endpoint)
    imports.update(signature.imports)
    body = _endpoint_body_src(chainlet.endpoint)
    imports.update(body.imports)

    src_parts = [
        f"class {chainlet.name}(stub.StubBase):",
        _indent(signature.src),
        _indent(body.src, 2),
        "\n",
    ]
    return _Source(src="\n".join(src_parts), imports=imports)


def _gen_stub_src_for_deps(
    dependencies: Iterable[definitions.ChainletAPIDescriptor],
) -> Optional[_Source]:
    """Generates a source code and imports for stub classes."""
    imports = set()
    src_parts = []
    for dep in dependencies:
        stub_src = _gen_stub_src(dep)
        imports.update(stub_src.imports)
        src_parts.append(stub_src.src)

    if not (imports or src_parts):
        return None
    return _Source(src="\n".join(src_parts), imports=imports)


# Truss Chainlet Gen ###################################################################


def _make_chainlet_dir(
    chain_name: str,
    chainlet_descriptor: definitions.ChainletAPIDescriptor,
    root: pathlib.Path,
) -> pathlib.Path:
    chainlet_name = chainlet_descriptor.name
    dir_name = f"chainlet_{chainlet_name}"
    chainlet_dir = root / definitions.GENERATED_CODE_DIR / chain_name / dir_name
    chainlet_dir.mkdir(exist_ok=True, parents=True)
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
    stubs = []
    stub_args = []
    for name, dep in chainlet_descriptor.dependencies.items():
        stubs.append(f"{name} = stub.factory({dep.name}, self._context, '{dep.name}')")
        stub_args.append(f"{name}={name}")

    if stub_args:
        init_args = f"context=self._context, {', '.join(stub_args)}"
    else:
        init_args = "context=self._context"

    user_chainlet_ref = _gen_chainlet_import_and_ref(chainlet_descriptor)
    imports.update(user_chainlet_ref.imports)
    body = _indent(
        "\n".join(
            [f"logging.info(f'Loading Chainlet `{chainlet_descriptor.name}`.')"]
            + stubs
            + [f"self._chainlet = {user_chainlet_ref.src}({init_args})"]
        )
    )
    src = "\n".join(["def load(self) -> None:", body])
    return _Source(src=src, imports=imports)


def _gen_predict_src(chainlet_descriptor: definitions.ChainletAPIDescriptor) -> _Source:
    """Generates AST for the `predict` method of the truss model."""
    if chainlet_descriptor.endpoint.is_generator:
        # TODO: implement generator.
        raise NotImplementedError("Generator.")

    imports = set()
    parts = []
    def_str = "async def" if chainlet_descriptor.endpoint.is_async else "def"
    parts.append(f"{def_str} predict(self, payload):")
    # Add error handling context manager:
    parts.append(
        _indent(
            f"with utils.exception_to_http_error("
            f'include_stack=True, chainlet_name="{chainlet_descriptor.name}"):'
        )
    )
    # Convert items from json payload dict to an arg-list, parsing pydantic models.
    args = []
    for arg_name, arg_type in chainlet_descriptor.endpoint.input_names_and_types:
        if arg_type.is_pydantic:
            type_ref = _gen_type_import_and_ref(arg_type)
            imports.update(type_ref.imports)
            args.append(f"{arg_name}={type_ref.src}.parse_obj(payload['{arg_name}'])")
        else:
            args.append(f"{arg_name}=payload['{arg_name}']")

    # Invoke Chainlet.
    maybe_await = "await " if chainlet_descriptor.endpoint.is_async else ""
    args_src = ",".join(args)
    run = chainlet_descriptor.endpoint.name
    parts.append(_indent(f"result = {maybe_await}self._chainlet.{run}({args_src})", 2))

    # Return as json tuple, serialize pydantic models.
    if len(chainlet_descriptor.endpoint.output_types) == 1:
        output_type = chainlet_descriptor.endpoint.output_types[0]
        result = "result.dict()" if output_type.is_pydantic else "result"
    else:
        result_parts = [
            f"result[{i}].dict()" if t.is_pydantic else f"result[{i}]"
            for i, t in enumerate(chainlet_descriptor.endpoint.output_types)
        ]
        result = f"{', '.join(result_parts)}"

    parts.append(_indent(f"return {result}"))

    return _Source(src="\n".join(parts), imports=imports)


def _gen_truss_chainlet_model(
    chainlet_descriptor: definitions.ChainletAPIDescriptor,
) -> _Source:
    logging.info(f"Generating Truss model for `{chainlet_descriptor.name}`.")
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

    if issubclass(chainlet_descriptor.user_config_type.raw, type(None)):
        userconfig_pin = "UserConfigT = type(None)"
    else:
        user_config_ref = _gen_type_import_and_ref(chainlet_descriptor.user_config_type)
        imports.update(user_config_ref.imports)
        userconfig_pin = f"UserConfigT = {user_config_ref.src}"
    return _Source(src=f"{userconfig_pin}\n\n{model_class_src}", imports=imports)


# Remote Chainlet Gen #################################################################


def _gen_truss_chainlet_file(
    chainlet_dir: pathlib.Path,
    chainlet_descriptor: definitions.ChainletAPIDescriptor,
    dependencies: Iterable[definitions.ChainletAPIDescriptor],
) -> pathlib.Path:
    """Generates code that wraps a Chainlet as a truss-compatible model."""
    file_path = chainlet_dir / truss_config.DEFAULT_MODEL_MODULE_DIR / _MODEL_FILENAME
    file_path.parent.mkdir(parents=True, exist_ok=True)
    (chainlet_dir / truss_config.DEFAULT_MODEL_MODULE_DIR / "__init__.py").touch()
    imports = set()
    src_parts = []
    if maybe_stub_src := _gen_stub_src_for_deps(dependencies):
        imports.update(maybe_stub_src.imports)
        src_parts.append(maybe_stub_src.src)

    model_src = _gen_truss_chainlet_model(chainlet_descriptor)
    src_parts.append(model_src.src)
    imports.update(model_src.imports)
    imports_str = "\n".join(imports)
    src_str = "\n".join(src_parts)
    file_path.write_text(f"{imports_str}\n{src_str}")
    _format_python_file(file_path)
    return file_path


# Truss Gen ############################################################################


def _copy_python_source_files(root_dir: pathlib.Path, dest_dir: pathlib.Path) -> None:
    """Copy all python files under root recursively, but skips pycache."""

    def python_files_only(path, names):
        return [name for name in names if name == "__pycache__"]

    shutil.copytree(root_dir, dest_dir, ignore=python_files_only, dirs_exist_ok=True)


def _make_truss_config(
    chainlet_dir: pathlib.Path,
    chains_config: definitions.RemoteConfig,
    user_config: definitions.UserConfigT,
    chainlet_to_service: Mapping[str, definitions.ServiceDescriptor],
    model_name: str,
) -> truss_config.TrussConfig:
    """Generate a truss config for a Chainlet."""
    config = truss_config.TrussConfig()
    config.model_name = model_name
    config.model_class_filename = _MODEL_FILENAME
    config.model_class_name = _MODEL_CLS_NAME
    # Compute.
    compute = chains_config.get_compute_spec()
    config.resources.cpu = str(compute.cpu_count)
    config.resources.accelerator = compute.accelerator
    config.resources.use_gpu = bool(compute.accelerator.count)
    # TODO: expose this setting directly.
    config.runtime.predict_concurrency = compute.cpu_count
    # Image.
    image = chains_config.docker_image
    config.base_image = truss_config.BaseImage(image=image.base_image)
    pip_requirements: list[str] = []
    if image.pip_requirements_file:
        pip_requirements.extend(
            req
            for req in pathlib.Path(image.pip_requirements_file.abs_path)
            .read_text()
            .splitlines()
            if not req.strip().startswith("#")
        )
    pip_requirements.extend(image.pip_requirements)
    # TODO: `pip_requirements` will add server requirements which give version
    #  conflicts. Check if that's still the case after relaxing versions.
    # config.requirements = pip_requirements
    pip_requirements_file_path = chainlet_dir / _REQUIREMENTS_FILENAME
    pip_requirements_file_path.write_text("\n".join(pip_requirements))
    # Absolute paths don't work with remote build.
    config.requirements_file = _REQUIREMENTS_FILENAME
    config.system_packages = image.apt_requirements
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
    # Metadata.
    chains_metadata: definitions.TrussMetadata = definitions.TrussMetadata(
        user_config=user_config, chainlet_to_service=chainlet_to_service
    )
    config.model_metadata[definitions.TRUSS_CONFIG_CHAINS_KEY] = chains_metadata.dict()
    config.write_to_yaml_file(
        chainlet_dir / serving_image_builder.CONFIG_FILE, verbose=True
    )
    return config


def gen_truss_chainlet(
    options: definitions.DeploymentOptions,
    chainlet_descriptor: definitions.ChainletAPIDescriptor,
    dependencies: Iterable[definitions.ChainletAPIDescriptor],
    chainlet_name_to_url: Mapping[str, str],
    chain_root: pathlib.Path,
    gen_root: pathlib.Path,
) -> pathlib.Path:
    # TODO: support file-based config (and/or merge file and python-src config values).
    remote_config = chainlet_descriptor.chainlet_cls.remote_config
    chainlet_name = remote_config.name or chainlet_descriptor.name
    # Filter needed services and customize options.
    dep_services = {}
    for dep in chainlet_descriptor.dependencies.values():
        dep_services[dep.name] = definitions.ServiceDescriptor(
            name=dep.name,
            predict_url=chainlet_name_to_url[dep.name],
            options=dep.options,
        )

    chainlet_dir = _make_chainlet_dir(options.chain_name, chainlet_descriptor, gen_root)

    _make_truss_config(
        chainlet_dir,
        remote_config,
        chainlet_descriptor.chainlet_cls.default_user_config,
        dep_services,
        f"{options.chain_name}.{chainlet_name}",
    )
    # TODO This assume all imports are absolute w.r.t chain root (or site-packages).
    _copy_python_source_files(
        chain_root, chainlet_dir / truss_config.DEFAULT_BUNDLED_PACKAGES_DIR
    )
    chainlet_file = _gen_truss_chainlet_file(
        chainlet_dir, chainlet_descriptor, dependencies
    )
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

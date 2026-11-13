import re

from truss.base.constants import TEMPLATES_DIR

# Every {{ }} expression in the Dockerfile templates, grouped by why it can't
# inject Dockerfile syntax. A new expression must be added under a reason, or
# escaped with one of the dockerfile_* filters in truss/util/jinja.py.
DOCKERFILE_TEMPLATE_EXPRESSION_ALLOWLIST: dict[str, list[str]] = {
    "Template literal or truss constant": [
        '"RUN --mount=type=secret,id=" + hf_access_token_file_name + ",dst=/etc/secrets/" + hf_access_token_file_name if use_hf_secret else "RUN"',
        "NONROOT_DEFAULT_USER_ID",
        "UV_VERSION",
        "app_username",
        "base_server_requirements_filename",
        "chown_and_switch_to_regular_user_if_enabled()",
        'chown_and_switch_to_regular_user_if_enabled(["/var/lib/nginx", "/var/log/nginx", "/run", "/dev/shm"])',
        "clean_uv_env",
        "config_requirements_filename",
        "constraints_filename",
        "control_server_dir",
        "copy_config_yaml()",
        "default_owner",
        "dir",
        "packages_dir",
        "proxy_config_path",
        "pyproject_toml_filename",
        "server_requirements_filename",
        "super()",
        "supervisor_config_path",
        "supervisor_server_url",
        "sys_pip_install_command",
        "system_packages_filename",
        "ubuntu_sources_list_file",
        "user_supplied_requirements_filename",
        "uv_lock_filename",
        "workspace_dir",
    ],
    "Build host setting, not model config": [
        "apt_cache_mount",
        "apt_mirror_url",
        "pip_cache_mount",
        "py_no_cache_dir",
        "uv_cache_mount",
    ],
    "Number, version, or hash computed by truss": [
        "control_python_version",
        "max_supported_python_minor_version_in_custom_base_image",
        "max_supported_python_version_in_custom_base_image",
        "min_supported_python_minor_version_in_custom_base_image",
        "min_supported_python_version_in_custom_base_image",
        "model_cache_hash",
        "run_as_user_id",
        "supported_python_major_version_in_custom_base_image",
        "truss_hash",
    ],
    "Checked by validate_dockerfile_rendered_fields": [
        "base_image_name_and_tag",
        "config.base_image.image",
        "config.base_image.python_executable_path",
        "config.bundled_packages_dir",
        "config.data_dir",
        "config.model_module_dir",
        "config.python_version",
        "credential",
        "env_var_name",
        "env_var_value",
        "path",
        "python_exec_path",
        "python_executable",
        "secret",
    ],
    "JSON string inside an exec-form CMD array": [
        "config.docker_server.start_command | tojson"
    ],
    "User-supplied build_commands, run as-is by design": ["command"],
}

_DOCKERFILE_ESCAPE_FILTER_RE = re.compile(
    r"\|\s*(dockerfile_env_value|dockerfile_shell_value|dockerfile_copy_path)$"
)


def test_dockerfile_template_expressions_are_allowlisted():
    found = set()
    for path in TEMPLATES_DIR.rglob("*Dockerfile*.jinja"):
        for expression in re.findall(
            r"\{\{-?\s*(.*?)\s*-?\}\}", path.read_text(), re.S
        ):
            normalized = " ".join(expression.split())
            if not _DOCKERFILE_ESCAPE_FILTER_RE.search(normalized):
                found.add(normalized)
    allowlisted = {
        expression
        for expressions in DOCKERFILE_TEMPLATE_EXPRESSION_ALLOWLIST.values()
        for expression in expressions
    }

    assert sorted(found - allowlisted) == [], (
        "New Dockerfile template expressions: add each under a reason in "
        "DOCKERFILE_TEMPLATE_EXPRESSION_ALLOWLIST, or escape it with a dockerfile_* filter."
    )
    assert sorted(allowlisted - found) == [], (
        "Allowlisted expressions no longer in any Dockerfile template; remove them."
    )

# Truss Development Guide

## Package Management

Use `uv` for all Python operations:
```bash
uv run pytest ...          # Run tests
uv run ruff check .        # Lint
uv run ruff format .       # Format
uv run truss <command>     # Run truss CLI
```

## Linting & Formatting

Always run before committing:
```bash
uv run ruff check <files>
uv run ruff format <files>
```

## Code Style

- **No inline imports** - Always place imports at the top of the file in global scope, never inside functions.

## Testing

```bash
# Unit tests
uv run pytest truss/tests/path/to/test.py -v

# Integration tests (requires Docker)
uv run pytest truss/tests/test_custom_server.py -v -m integration

# Run specific test
uv run pytest truss/tests/test_config.py::test_name -v
```

**Test fixtures:** Use `test_data_path` fixture (from `conftest.py`) for config files in `truss/tests/test_data/`.

## Key Code Locations

| Area | Location |
|------|----------|
| Config validation | `truss/base/truss_config.py` |
| TRT-LLM config | `truss/base/trt_llm_config.py` |
| Image building | `truss/contexts/image_builder/serving_image_builder.py` |
| Templates | `truss/templates/` |
| CLI | `truss/cli/cli.py` |
| Remote/push | `truss/remote/baseten/remote.py` |

## Architecture Notes

- `docker_server` deployments use supervisord to manage processes
- Config is validated via Pydantic models in `truss_config.py`
- Image build generates Dockerfile from Jinja templates

## Cursor Cloud specific instructions

- The update script installs `uv` and runs `uv sync`. After it completes, all dev dependencies are available via `uv run`.
- **Lint/format**: `uv run ruff check .` and `uv run ruff format --check .` (see `Makefile` `format` target for auto-fix).
- **Unit tests**: `uv run pytest truss/tests -m "not integration" -x -q` runs ~950 tests. Running the full suite at once may OOM on small VMs; run subsets or individual test modules if needed (e.g. `uv run pytest truss/tests/test_config.py -v`).
- **Chains/Train tests**: `uv run pytest truss-chains/tests/ -q` and `uv run pytest truss-train/tests/ -q`.
- **Integration tests** require Docker and are marked with `-m integration`. Docker is not available in the default Cloud Agent VM.
- **CLI**: `uv run truss --help` to verify the CLI works. `uv run truss init <name>` scaffolds a new model project (use piped input for non-interactive: `echo "model-name" | uv run truss init <dir>`).
- **mypy**: `uv run mypy <file>` for type checking.
- The test in `truss/tests/templates/control/control/test_server.py::test_patch_model_code_update_predict_on_long_load_time` is a known flaky test (timing-sensitive); failures there are pre-existing and not caused by your changes.

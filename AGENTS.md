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

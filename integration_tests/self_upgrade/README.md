# Self-Upgrade Integration Tests

This directory contains integration tests for verifying the `truss upgrade` functionality
across different installation methods using Docker containers.

## Environments Tested

- **uv**: Tests `uv tool install` and `uv pip install` methods
- **pipx**: Tests `pipx install` method
- **conda**: Tests installation in a conda environment

## Running the Tests

```bash
cd integration_tests/self_upgrade
./run_tests.sh

# With a specific version:
TRUSS_VERSION=0.12.6rc507 ./run_tests.sh

# Test specific environments:
ENVIRONMENTS="uv_tool uv_venv" TRUSS_VERSION=0.12.6rc507 ./run_tests.sh
```

## What's Tested

### Upgrade Tests (`test_in_container.py`)
1. **Installation detection**: Verifies `detect_installation_method()` returns the correct method
2. **Upgrade command**: Verifies the correct upgrade command is generated
3. **Settings integration**: Tests with `check_for_updates` enabled and disabled
4. **Actual upgrade**: Performs a real upgrade to latest

### Settings TOML Tests (`test_settings_toml.py`)
1. **Default creation**: When no settings file exists, creates one with `check_for_updates = true`
2. **Read true**: Correctly reads `check_for_updates = true` from file
3. **Read false**: Correctly reads `check_for_updates = false` from file
4. **Notify respects setting**: `notify_if_outdated()` is silent when disabled
5. **Correct location**: Settings written to `~/.config/truss/settings.toml`

## Running Settings TOML Tests

```bash
cd integration_tests/self_upgrade

# Build and run settings tests (uses uv_venv environment)
docker build --no-cache -f Dockerfile.uv --build-arg INSTALL_METHOD=venv -t truss-test-settings . && \
docker run --rm truss-test-settings python /tests/test_settings_toml.py
```

## Notes

- Tests both settings file configurations (check_for_updates: true/false)
- Containers are ephemeral and cleaned up after tests

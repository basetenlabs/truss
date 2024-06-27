import pytest

from truss.errors import ValidationError
from truss.validation import (
    validate_cpu_spec,
    validate_memory_spec,
    validate_secret_name,
)


@pytest.mark.parametrize(
    "secret_name, should_error",
    [
        (None, True),
        (1, True),
        ("", True),
        (".", True),
        ("..", True),
        ("a" * 253, False),
        ("a" * 254, True),
        ("-", False),
        ("-.", False),
        ("a-.", False),
        ("-.a", False),
        ("a-foo", False),
        ("a.foo", False),
        (".foo", False),
        ("x\\", True),
        ("a_b", False),
        ("_a", False),
        ("a_", False),
    ],
)
def test_validate_secret_name(secret_name, should_error):
    does_error = False
    try:
        validate_secret_name(secret_name)
    except:  # noqa
        does_error = True

    assert does_error == should_error


@pytest.mark.parametrize(
    "cpu_spec, expected_valid",
    [
        (None, False),
        ("", False),
        ("1", True),
        ("1.5", True),
        ("1.5m", True),
        (1, False),
        ("1m", True),
        ("1M", False),
        ("M", False),
        ("M1", False),
    ],
)
def test_validate_cpu_spec(cpu_spec, expected_valid):
    if not expected_valid:
        with pytest.raises(ValidationError):
            validate_cpu_spec(cpu_spec)
    else:
        validate_cpu_spec(cpu_spec)


@pytest.mark.parametrize(
    "mem_spec, expected_valid, memory_in_bytes",
    [
        (None, False, None),
        (1, False, None),
        ("1m", False, None),
        ("1k", True, 10**3),
        ("512k", True, 512 * 10**3),
        ("512M", True, 512 * 10**6),
        ("1.5Gi", True, 1.5 * 1024**3),
        ("abc", False, None),
        ("1024", True, 1024),
    ],
)
def test_validate_mem_spec(mem_spec, expected_valid, memory_in_bytes):
    if not expected_valid:
        with pytest.raises(ValidationError):
            validate_memory_spec(mem_spec)
    else:
        assert memory_in_bytes == validate_memory_spec(mem_spec)

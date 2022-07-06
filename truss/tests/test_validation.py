import pytest

from truss.validation import validate_secret_name


@pytest.mark.parametrize('secret_name, should_error', [
    (None, True),
    (1, True),
    ('', True),
    ('.', True),
    ('..', True),
    ('a' * 253, False),
    ('a' * 254, True),
    ('-', False),
    ('-.', False),
    ('a-.', False),
    ('-.a', False),
    ('a-foo', False),
    ('a.foo', False),
    ('.foo', False),
    ('x\\', True),
    ('a_b', False),
    ('_a', False),
    ('a_', False),
])
def test_validate_secret_name(secret_name, should_error):
    does_error = False
    try:
        validate_secret_name(secret_name)
    except:  # noqa
        does_error = True

    assert does_error == should_error

import pytest

from truss.util.jinja import dockerfile_env_value


def test_plain_value_is_double_quoted():
    assert dockerfile_env_value("python main.py --port 8000") == (
        '"python main.py --port 8000"'
    )


def test_single_quotes_kept_verbatim():
    # tojson escaped these to \u0027, which the Dockerfile ENV parser keeps
    # verbatim; the filter must keep them as-is instead.
    value = "vllm serve --hf-overrides '{\"a\": 1}'"
    assert dockerfile_env_value(value) == (
        '"vllm serve --hf-overrides \'{\\"a\\": 1}\'"'
    )


def test_html_sensitive_chars_kept_verbatim():
    assert dockerfile_env_value("a < b > c & d") == '"a < b > c & d"'


def test_dollar_escaped_to_defer_expansion_to_runtime():
    assert dockerfile_env_value("serve --host $HOST") == '"serve --host \\$HOST"'


def test_backslash_and_double_quote_escaped():
    assert dockerfile_env_value('a\\b "c"') == '"a\\\\b \\"c\\""'


@pytest.mark.parametrize("value", ["line1\nline2", "line1\rline2", "line1\r\nline2"])
def test_line_breaks_rejected_without_echoing_value(value):
    with pytest.raises(ValueError, match="line break") as exc_info:
        dockerfile_env_value(value)
    assert "line1" not in str(exc_info.value)

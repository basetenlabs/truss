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


def test_newline_rejected():
    with pytest.raises(ValueError, match="newline"):
        dockerfile_env_value("line1\nline2")

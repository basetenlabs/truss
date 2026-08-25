import os
import shlex
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from truss.util.jinja import dockerfile_env_value, dockerfile_shell_value


def _posix_sh_curl_argv(quote, url: str) -> tuple[list[str], bool, str]:
    """Run `curl -L <quoted url> -o <quoted dest>` under /bin/sh with a stub curl.

    Returns (curl argv, whether a canary file the payload tried to write exists).
    """
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        argv_file = tmp_path / "argv"
        pwned = tmp_path / "pwned"
        curl = tmp_path / "curl"
        curl.write_text(f"#!/bin/sh\nprintf '%s\\0' \"$@\" > '{argv_file}'\n")
        curl.chmod(0o755)
        env = os.environ.copy()
        env["PATH"] = f"{tmp_path}:{env['PATH']}"
        url = url.replace("PWNED", str(pwned))
        cmd = f"curl -L {quote(url)} -o {quote(str(tmp_path / 'out'))}"
        subprocess.run(["/bin/sh", "-c", cmd], env=env, check=True, capture_output=True)
        argv = [a.decode() for a in argv_file.read_bytes().split(b"\0") if a]
        return argv, pwned.exists(), url


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


def test_shell_injection_metacharacters_in_url():
    url = 'http://example.com/file" ; echo pwned ; echo "'
    quoted = dockerfile_shell_value(url)
    assert quoted == shlex.quote(url)
    assert quoted.startswith("'") and quoted.endswith("'")


def test_shell_value_quotes_backticks():
    url = "http://example.com/`id`"
    quoted = dockerfile_shell_value(url)
    assert quoted == shlex.quote(url)
    assert quoted == "'http://example.com/`id`'"


@pytest.mark.skipif(sys.platform == "win32", reason="/bin/sh is not on Windows CI")
@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/model.bin",
        'http://example.com/x" ; touch PWNED ; echo "',
        "http://example.com/`touch PWNED`",
        "http://example.com/$(touch PWNED)",
        "http://example.com/$HOME/weights.bin",
        "http://example.com/${HOME}",
        "http://example.com/file#token",
        "http://example.com/it's.bin",
        "http://example.com/a b.bin",
    ],
)
def test_shell_value_is_one_posix_sh_word(url):
    argv, pwned, resolved_url = _posix_sh_curl_argv(dockerfile_shell_value, url)
    assert argv == ["-L", resolved_url, "-o", argv[3]]
    assert pwned is False


@pytest.mark.skipif(sys.platform == "win32", reason="/bin/sh is not on Windows CI")
def test_env_filter_still_executes_backticks_in_posix_sh():
    """Cretz's review: ENV quoting leaves backticks live when the value is in RUN."""
    argv, pwned, _resolved = _posix_sh_curl_argv(
        dockerfile_env_value, "http://example.com/`touch PWNED`"
    )
    assert pwned is True
    assert argv[1] == "http://example.com/"


@pytest.mark.parametrize("value", ["line1\nline2", "line1\rline2", "line1\r\nline2"])
def test_shell_value_line_breaks_rejected_without_echoing_value(value):
    with pytest.raises(ValueError, match="line break") as exc_info:
        dockerfile_shell_value(value)
    assert "line1" not in str(exc_info.value)

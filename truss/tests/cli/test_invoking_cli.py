import pytest

from truss.cli.utils import invoking_cli


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, "truss"),
        ("", "truss"),
        ("  ", "truss"),
        ("truss", "truss"),
        ("other", "truss"),
        ("baseten", "baseten"),
        ("BASETEN", "baseten"),
        ("  Baseten  ", "baseten"),
    ],
)
def test_invoking_cli_reads_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(invoking_cli.INVOKING_CLI_ENV, raising=False)
    else:
        monkeypatch.setenv(invoking_cli.INVOKING_CLI_ENV, raw)
    assert invoking_cli.invoking_cli() == expected


@pytest.mark.parametrize(
    "builder,job_id,truss_cmd,baseten_cmd",
    [
        (
            invoking_cli.train_logs,
            "job-1",
            "truss train logs --job-id job-1 --tail",
            "baseten train job logs --job-id job-1 --tail",
        ),
        (
            invoking_cli.train_metrics,
            "job-1",
            "truss train metrics --job-id job-1",
            "baseten train job metrics --job-id job-1",
        ),
        (
            invoking_cli.train_view,
            "job-1",
            "truss train view --job-id=job-1",
            "baseten train job describe --job-id job-1",
        ),
        (
            invoking_cli.train_stop,
            "job-1",
            "truss train stop --job-id job-1",
            "baseten train job stop --job-id job-1",
        ),
    ],
)
def test_follow_up_commands(monkeypatch, builder, job_id, truss_cmd, baseten_cmd):
    monkeypatch.delenv(invoking_cli.INVOKING_CLI_ENV, raising=False)
    assert builder(job_id) == truss_cmd
    monkeypatch.setenv(invoking_cli.INVOKING_CLI_ENV, "baseten")
    assert builder(job_id) == baseten_cmd


def test_train_cache_summarize(monkeypatch):
    monkeypatch.delenv(invoking_cli.INVOKING_CLI_ENV, raising=False)
    assert (
        invoking_cli.train_cache_summarize("my-project")
        == 'truss train cache summarize "my-project"'
    )
    monkeypatch.setenv(invoking_cli.INVOKING_CLI_ENV, "baseten")
    assert (
        invoking_cli.train_cache_summarize("my-project")
        == "baseten train project cache describe --project my-project"
    )


def test_ssh_setup(monkeypatch):
    monkeypatch.delenv(invoking_cli.INVOKING_CLI_ENV, raising=False)
    assert invoking_cli.ssh_setup() == "truss ssh setup"
    monkeypatch.setenv(invoking_cli.INVOKING_CLI_ENV, "baseten")
    assert invoking_cli.ssh_setup() == "baseten ssh setup"

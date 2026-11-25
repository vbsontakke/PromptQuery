import os

import pytest

from eval.atlas import run


def test_missing_configuration_is_a_clear_error_not_a_traceback(monkeypatch, capsys):
    for k in ("PQ_DSN", "PQ_TENANT"):
        monkeypatch.delenv(k, raising=False)
    assert run.main(["--split", "dev"]) == 2
    assert "configuration error" in capsys.readouterr().err


@pytest.mark.skipif(not os.environ.get("PQ_TEST_BOT_DSN"), reason="needs a seeded Atlas DB")
def test_missing_api_key_is_a_clear_error_not_a_traceback(monkeypatch, capsys):
    monkeypatch.setenv("PQ_DSN", os.environ["PQ_TEST_BOT_DSN"])
    monkeypatch.setenv("PQ_ADMIN_DSN", os.environ["PQ_TEST_OWNER_DSN"])
    monkeypatch.setenv("PQ_TENANT", "acme")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert run.main(["--ids", "fin_cash_position"]) == 2
    err = capsys.readouterr().err
    assert "cannot start the evaluation" in err and "ANTHROPIC_API_KEY" in err and "--oracle" in err


def test_unknown_scenario_selection_is_reported(monkeypatch, capsys):
    monkeypatch.setenv("PQ_DSN", "postgresql://x")
    monkeypatch.setenv("PQ_TENANT", "acme")
    assert run.main(["--ids", "no_such_scenario"]) == 2
    assert "no scenarios selected" in capsys.readouterr().err

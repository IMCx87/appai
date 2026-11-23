"""Testes do app Streamlit inteiro (AppTest roda o script sem navegador)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setenv("APPAI_DATA_DIR", str(tmp_path))


def test_app_renders_all_tabs_without_errors():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    labels = [t.label for t in at.tabs]
    for name in ("💬 Chat", "📊 Benchmark", "🧊 Cold start", "📚 RAG", "💰 Custo"):
        assert name in labels


def test_sessions_do_not_share_state(mock):
    """Regressão: defaults mutáveis (listas) eram compartilhados entre sessões."""
    cfg, _, _ = mock
    a = AppTest.from_file(APP, default_timeout=30).run()
    a.session_state["api_url"] = cfg["base"]
    a.session_state["auth_type"] = "None / No Auth"
    a.session_state["model_name"] = "mock-llm"
    a.chat_input[0].set_value("olá").run()
    assert not a.exception
    assert len(a.session_state["messages"]) == 2

    b = AppTest.from_file(APP, default_timeout=30).run()
    assert b.session_state["messages"] == []
    assert b.session_state["bench_levels"] is not a.session_state["bench_levels"]


def test_benchmark_runs_end_to_end(mock):
    cfg, _, root = mock
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.session_state["api_url"] = cfg["base"]
    at.session_state["auth_type"] = "None / No Auth"
    at.session_state["model_name"] = "mock-llm"
    at.session_state["bench_levels"] = [1, 2]
    at.session_state["bench_rounds"] = 1
    at.session_state["bench_out"] = 16
    at.session_state["tele_vllm_url"] = root + "/metrics"
    at.run()
    at.button(key="run_bench").click().run()
    assert not at.exception
    res = at.session_state["bench_result"]
    assert res and [lv["load"] for lv in res["data"]["levels"]] == [1, 2]
    assert res["id"]  # salvo no histórico

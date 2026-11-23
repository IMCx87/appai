"""Custo, histórico, cold start, RAG e relatório."""

import time

from appai import cost, rag, report
from appai.bench import DEFAULT_SLO, run_sweep
from appai.coldstart import load_throughput_gbs, measure_cold_start
from appai.store import Store, display_name


def base_params(**kw):
    p = {"peak_tps": 1000, "utilization": 0.5, "gpu_hour": 2.0, "gpu_count": 2,
         "monthly_in_m": 2000, "monthly_out_m": 1000, "price_in": 1.0, "price_out": 4.0,
         "energy_separate": False}
    p.update(kw)
    return p


def test_onprem_analysis_numbers():
    a = cost.onprem_analysis(base_params())
    assert a["set_month"] == 2.0 * 2 * 730
    assert abs(a["capacity_out_m"] - 1000 * 3600 * 730 * 0.5 / 1e6) < 1e-9  # 1314M
    assert a["replicas"] == 1
    assert a["cloud_per_m_out"] == 4.0 + 2.0 * 1.0
    assert a["cloud_month"] == 2000 * 1 + 1000 * 4
    assert abs(a["breakeven_m"] - 2920 / 6.0) < 1e-9 and a["breakeven_within_capacity"]
    big = cost.onprem_analysis(base_params(monthly_out_m=3000))
    assert big["replicas"] == 3 and big["onprem_month"] == 3 * 2920


def test_energy_and_curve():
    e = cost.onprem_analysis(base_params(energy_separate=True, power_w=500, pue=1.5,
                                         kwh_price=0.2))
    assert abs(e["energy_month"] - 500 * 2 * 1.5 * 730 / 1000 * 0.2) < 1e-9
    curve = cost.cost_curve(base_params())
    assert curve[0]["cloud"] == 0 and curve[-1]["onprem"] >= curve[0]["onprem"]
    assert cost.energy_per_m_tokens(2.0, 1_000_000) == 2.0


def test_store_roundtrip(tmp_path):
    s = Store(tmp_path / "x" / "h.db")
    assert s.enabled
    rid = s.save("sweep", "L40S", {"endpoint": "e", "model": "m"}, {"levels": [1]})
    assert s.list()[0]["id"] == rid and s.list("rag") == []
    s.rename(rid, "H100")
    got = s.get(rid)
    assert got["label"] == "H100" and got["data"] == {"levels": [1]}
    assert "H100" in display_name(got)
    s.delete(rid)
    assert s.get(rid) is None


def test_store_disabled_when_unwritable(tmp_path):
    f = tmp_path / "file"
    f.write_text("x")
    s = Store(f / "sub" / "h.db")
    assert not s.enabled and s.save("sweep", "", {}, {}) is None and s.list() == []


def test_cold_start_against_hibernated_mock(mock):
    cfg, sim, _ = mock
    sim.hibernate()
    events = []
    r = measure_cold_start(cfg, timeout_s=10, poll_s=0.2, on_event=events.append)
    assert r["ok"] and r["attempts"] >= 2
    assert any(e["status"] == 404 for e in events)
    assert 0.5 <= r["ready_s"] < 5 and r["total_s"] >= r["ready_s"]
    assert load_throughput_gbs(16, 8) == 2.0 and load_throughput_gbs(0, 8) is None


def test_rag_pipeline_against_mock(mock):
    cfg, _, _ = mock
    idx = rag.RagIndex()
    stats = idx.build(cfg, "mock-embed", rag.SAMPLE_CORPUS)
    assert stats["chunks"] >= len(rag.SAMPLE_CORPUS) and stats["dim"] == 256
    one = rag.run_query(cfg, cfg, "mock-embed", idx, "O que acontece quando o KV cache enche?",
                        3, 16, keep_text=True)
    assert one["ok"] and one["sources"][0] == "KV cache"
    assert one["user_ttft_ms"] >= one["prefill_ms"]
    recs = rag.run_rag_bench(cfg, cfg, "mock-embed", idx, rag.SAMPLE_QUESTIONS, 6, 3, 3, 8,
                             lambda *a: None)
    s = rag.summarize_rag(recs)
    assert s["errors"] == 0 and s["stages"][0]["p50 (ms)"] is not None
    assert len(rag.chunk_text(" ".join(["w"] * 250), 120, 20)) == 3


def test_html_report(mock):
    cfg, _, _ = mock
    spec = {"question": "Q", "in": 20, "out": 8, "dist": "Fixo", "nocache": True}
    res = run_sweep(cfg, {"mode": "closed", "levels": [1, 2], "rounds": 1, "spec": spec},
                    lambda *a: None)
    run = {"label": "teste", "info": {"mode": "closed", "model": "mock-llm",
                                      "timestamp": time.strftime("%F")},
           "data": {"levels": res["levels"]}}
    html = report.sweep_report(run, DEFAULT_SLO, "Relatório X")
    assert html.startswith("<!doctype html>") and "Relatório X" in html
    assert "Capacidade dentro do SLO" in html and "plotly" in html.lower()

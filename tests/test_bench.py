import random

from appai import bench
from appai.client import fetch_models, sanitize_messages, stream_collect

SLO = {"ttft_ms": 1000.0, "itl_ms": 50.0, "max_err_pct": 1.0}


def rec(ttft=0.1, itl=10.0, ok=True, out=10, tps=50.0, latency=0.5):
    return {"ok": ok, "ttft": ttft, "itl": itl, "out": out, "in": 5, "tps": tps,
            "latency": latency, "cost": 0.0, "retries": 0, "lag": 0.0}


def test_pct_interpolates():
    assert bench.pct([], 50) is None
    assert bench.pct([10], 95) == 10
    assert bench.pct([1, 2, 3, 4, 5], 50) == 3
    assert abs(bench.pct([0, 10], 95) - 9.5) < 1e-9


def test_summarize_and_slo():
    records = [rec(ttft=0.2)] * 9 + [rec(ttft=2.0)]
    row = bench.summarize(4, records, wall=2.0, slo=SLO)
    assert row["requests"] == 10 and row["errors"] == 0
    assert row["slo_ok_pct"] == 90.0
    assert row["goodput_rps"] == 4.5
    assert row["throughput"] == 50.0
    assert row["ttft_p95"] > 1000 and row["slo_pass"] is False
    row2 = bench.summarize(1, [rec()] * 5, wall=1.0, slo=SLO)
    assert row2["slo_pass"] is True


def test_errors_break_slo():
    row = bench.summarize(2, [rec()] * 9 + [rec(ok=False)], 1.0, SLO)
    assert row["err_pct"] == 10.0 and row["slo_pass"] is False


def test_capacity_requires_contiguous_passes():
    rows = [{"load": 1, "slo_pass": True, "goodput_rps": 1, "throughput": 10},
            {"load": 4, "slo_pass": True, "goodput_rps": 3, "throughput": 30},
            {"load": 8, "slo_pass": False, "goodput_rps": 2, "throughput": 35},
            {"load": 16, "slo_pass": True, "goodput_rps": 2, "throughput": 36}]
    cap = bench.capacity(rows)
    assert cap["max_ok"] == 4 and cap["first_fail"] == 8 and not cap["all_pass"]
    assert bench.capacity([{"load": 1, "slo_pass": False, "goodput_rps": 0,
                            "throughput": 1}])["max_ok"] is None


def test_arrival_offsets_mean_rate():
    offs = bench.arrival_offsets(4000, 10.0, random.Random(1))
    assert offs[0] == 0 and offs == sorted(offs)
    assert abs(offs[-1] / len(offs) - 0.1) < 0.01


def test_sanitize_messages():
    msgs = [{"role": "assistant", "content": "a"}, {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"}, {"role": "system", "content": "s"}]
    assert sanitize_messages(msgs) == [{"role": "user", "content": "u1\nu2"}]
    assert msgs[1]["content"] == "u1"  # não muta a entrada


def test_stream_against_mock(mock):
    cfg, sim, _ = mock
    assert "mock-llm" in fetch_models(cfg)
    text, meta = stream_collect(cfg, [{"role": "user", "content": "olá"}], 20)
    assert text and meta["out"] == 20 and meta["usage_reported"]
    assert meta["ttft"] > 0 and meta["itl"] > 0 and meta["tps"] > 0


def test_sweep_closed_and_open(mock):
    cfg, sim, _ = mock
    spec = {"question": "Q?", "in": 40, "out": 12, "dist": "Fixo", "cv": 0.5,
            "nocache": True, "dataset": None}
    seen = []
    res = bench.run_sweep(cfg, {"mode": "closed", "levels": [1, 4], "rounds": 2,
                                "spec": spec, "warmup": True},
                          lambda d, t, lv: seen.append((d, t)))
    assert [lv["load"] for lv in res["levels"]] == [1, 4]
    assert sum(len(lv["records"]) for lv in res["levels"]) == 10
    assert all(r["ok"] for lv in res["levels"] for r in lv["records"])
    assert seen[-1] == (10, 10) and len(res["warmup"]) == 2
    rows = bench.summarize_run(res["levels"], SLO)
    assert rows[0]["slo_pass"]

    res2 = bench.run_sweep(cfg, {"mode": "open", "levels": [5.0], "duration_s": 1,
                                 "spec": spec}, lambda *a: None)
    assert len(res2["levels"][0]["records"]) == 5


def test_force_output_param_reaches_server(mock):
    cfg, _, _ = mock
    cfg = {**cfg, "extra": {"ignore_eos": True}}
    r = bench.run_one(cfg, {"prompt": "x", "max_tokens": 200})
    assert r["ok"] and r["out"] == 200
    r2 = bench.run_one({**cfg, "extra": {}}, {"prompt": "x", "max_tokens": 200})
    assert r2["out"] == 120  # tamanho "natural" do mock


def test_errors_are_records_not_exceptions():
    cfg = {"base": "http://127.0.0.1:9/v1", "headers": {}, "verify": False, "model": "m",
           "temperature": 0, "max_retries": 1, "prices": (0, 0)}
    r = bench.run_one(cfg, {"prompt": "x", "max_tokens": 5})
    assert r["ok"] is False and "error" in r


def test_mock_accepts_many_simultaneous_connections():
    """Regressão: com o backlog padrão (5) do socketserver, 32 conexões
    simultâneas sofriam retransmissão TCP (~1 s) e o TTFT ficava falso."""
    from appai.mock_server import serve
    from tests.conftest import free_port
    port = free_port()
    srv, _ = serve(port, decode_tps=400, prefill_tps=50000, max_batch=64)
    cfg = {"base": f"http://127.0.0.1:{port}/v1", "headers": {}, "verify": False,
           "model": "mock-llm", "temperature": 0, "max_retries": 1, "prices": (0, 0)}
    try:
        reqs = [{"prompt": "x", "max_tokens": 8}] * 32
        records, _ = bench.run_closed_level(cfg, 32, reqs, lambda *a: None)
        ttft = [r["ttft"] for r in records if r["ok"]]
        assert len(ttft) == 32 and bench.pct(ttft, 95) < 0.8
    finally:
        srv.shutdown()

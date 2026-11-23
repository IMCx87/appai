import time

from appai import telemetry as t
from appai.bench import summarize

PROM = """
# HELP vllm:num_requests_running x
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="a",engine="0"} 3
vllm:num_requests_running{model_name="a",engine="1"} 2
vllm:num_requests_waiting{model_name="a"} 4
vllm:gpu_cache_usage_perc{model_name="a",engine="0"} 0.5
vllm:gpu_cache_usage_perc{model_name="a",engine="1"} 0.93
vllm:num_preemptions_total{model_name="a"} 7
garbage line
"""


def test_parse_and_extract_with_legacy_names():
    parsed = t.parse_prom_text(PROM)
    assert parsed["vllm:num_requests_running"][0][0] == {"model_name": "a", "engine": "0"}
    s = t.extract_vllm(parsed)
    assert s["running"] == 5 and s["waiting"] == 4
    assert s["kv"] == 0.93 and s["preempt"] == 7 and s["prefix_hits"] is None


def test_parse_queries():
    q = t.parse_prom_queries("GPU | avg(x)\n# c\n\nsum(y)")
    assert q == [("GPU", "avg(x)"), ("sum(y)", "sum(y)")]


def test_aggregate_windows_and_energy():
    levels = [{"load": 1, "t0": 10, "t1": 20}, {"load": 4, "t0": 20, "t1": 30}]
    samples = [{"t": float(x), "waiting": 0 if x < 20 else 3, "running": 1, "kv": x / 40,
                "preempt": 0 if x < 25 else 2, "prefix_hits": x, "prefix_queries": 2 * x}
               for x in range(8, 33)]
    power = [(float(x), 300.0) for x in range(10, 31)]
    agg = t.aggregate_levels(levels, samples, {"GPU power (W)": power})
    assert agg[0]["waiting_max"] == 0 and agg[1]["waiting_max"] == 3
    assert agg[1]["preempt_delta"] == 2 and abs(agg[0]["prefix_hit_rate"] - 0.5) < 1e-9
    assert agg[1]["prom:GPU power (W):mean"] == 300
    kwh = t.energy_kwh(t.find_power_series({"GPU power (W)": power}), 10, 30)
    assert abs(kwh - 300 * 20 / 3.6e6) < 1e-12


def _row(load, ttft, thr, slo_pass):
    return {"load": load, "slo_pass": slo_pass, "ttft_p95": ttft, "itl_p95": 20,
            "err_pct": 0, "throughput": thr, "lag_p95": 0}


def test_diagnose_is_hedged_and_finds_queue_kv_plateau():
    rows = [_row(4, 300, 100, True), _row(8, 900, 180, True), _row(16, 2500, 190, False)]
    infra = [{"load": 4}, {"load": 8, "kv_max": 0.5},
             {"load": 16, "waiting_max": 6, "kv_max": 0.97, "preempt_delta": 3,
              "prom:GPU util (%):mean": 95}]
    f = t.diagnose(rows, infra)
    kinds = {x["kind"] for x in f}
    assert {"slo", "kv", "preempt", "plateau"} <= kinds
    slo_txt = next(x["text"] for x in f if x["kind"] == "slo")
    assert "pode estar relacionado" in slo_txt and "fila" in slo_txt
    assert "causa" not in " ".join(x["text"] for x in f).lower()


def test_sampler_and_prometheus_against_mock(mock):
    cfg, sim, root = mock
    s = t.VllmSampler(root + "/metrics", interval=0.2).start()
    time.sleep(0.7)
    samples = s.stop()
    assert len(samples) >= 2 and s.errors == 0
    assert samples[0]["running"] == 0 and samples[0]["kv"] == 0
    now = time.time()
    time.sleep(1.1)
    res = t.collect_prometheus(root, t.parse_prom_queries(t.DEFAULT_PROM_QUERIES),
                               now - 2, time.time(), 1, False)
    assert not res["errors"] and res["series"]["GPU util (%)"]
    assert summarize(1, [], 1.0)["requests"] == 0

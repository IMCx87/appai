"""Motor de benchmark: closed-loop (concorrência fixa) e open-loop (Poisson),
resumo por nível, SLO e goodput.

Os registros por requisição são a fonte da verdade: o resumo e a capacidade
dentro do SLO são recalculados a partir deles, então mudar o SLO na tela não
exige rodar de novo.
"""

import concurrent.futures as cf
import random
import statistics
import time

from appai.client import chat_body, finalize_metrics, post_with_retry, stream_response
from appai.workload import make_requests

MODES = {
    "closed": "Concorrência fixa (closed-loop)",
    "open": "Taxa de chegada (open-loop, Poisson)",
}
MAX_INFLIGHT = 256

DEFAULT_SLO = {"ttft_ms": 1000.0, "itl_ms": 50.0, "max_err_pct": 1.0}


def run_one(cfg: dict, req: dict, scheduled: float | None = None) -> dict:
    """Uma requisição de benchmark (streaming). Nunca lança: erros viram registro."""
    body = chat_body(cfg, [{"role": "user", "content": req["prompt"]}],
                     req["max_tokens"], stream=True)
    t_wall = time.time()
    metrics = {"start": time.perf_counter()}
    lag = (metrics["start"] - scheduled) if scheduled is not None else 0.0
    retries = 0
    try:
        resp, retries = post_with_retry(cfg, body, stream=True)
        for _ in stream_response(resp, metrics):
            pass
        meta = finalize_metrics(metrics, cfg["model"], cfg["prices"])
        meta.update(ok=True, retries=retries,
                    latency=metrics["end"] - metrics["start"])
    except Exception as e:  # noqa: BLE001 — qualquer falha é um dado do benchmark
        meta = {"ok": False, "error": str(e)[:200], "retries": retries,
                "latency": time.perf_counter() - metrics["start"]}
    meta["t_start"] = t_wall
    meta["t_end"] = t_wall + meta["latency"]
    meta["lag"] = max(lag, 0.0)
    meta["max_tokens"] = req["max_tokens"]
    return meta


def pct(values: list, p: float):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def request_meets_slo(r: dict, slo: dict) -> bool:
    if not r.get("ok") or r.get("ttft") is None:
        return False
    if r["ttft"] * 1000 > slo["ttft_ms"]:
        return False
    return r.get("itl") is None or r["itl"] <= slo["itl_ms"]


def _r(x, nd=1):
    return None if x is None else round(x, nd)


def summarize(load: float, records: list, wall: float, slo: dict | None = None) -> dict:
    slo = slo or DEFAULT_SLO
    ok = [r for r in records if r.get("ok")]
    ttft = [r["ttft"] * 1000 for r in ok if r.get("ttft") is not None]
    itl = [r["itl"] for r in ok if r.get("itl")]
    tps = [r["tps"] for r in ok if r.get("tps")]
    e2e = [r["latency"] * 1000 for r in ok]
    lag = [r.get("lag", 0.0) * 1000 for r in records]
    out_tok = sum(r.get("out", 0) for r in ok)
    n = len(records)
    good = [r for r in records if request_meets_slo(r, slo)]
    err_pct = (n - len(ok)) / n * 100 if n else 0.0
    row = {
        "load": load,
        "requests": n,
        "errors": n - len(ok),
        "err_pct": round(err_pct, 2),
        "retries": sum(r.get("retries", 0) for r in records),
        "ttft_p50": _r(pct(ttft, 50)),
        "ttft_p95": _r(pct(ttft, 95)),
        "ttft_p99": _r(pct(ttft, 99)),
        "itl_mean": _r(statistics.fmean(itl)) if itl else None,
        "itl_p95": _r(pct(itl, 95)),
        "tps_mean": _r(statistics.fmean(tps)) if tps else None,
        "e2e_p50": _r(pct(e2e, 50), 0),
        "e2e_p95": _r(pct(e2e, 95), 0),
        "throughput": _r(out_tok / wall) if wall > 0 else None,
        "rps": round(len(ok) / wall, 2) if wall > 0 else None,
        "goodput_rps": round(len(good) / wall, 2) if wall > 0 else None,
        "slo_ok_pct": round(len(good) / n * 100, 1) if n else None,
        "tokens_in": sum(r.get("in", 0) for r in ok),
        "tokens_out": out_tok,
        "cost": round(sum(r.get("cost", 0.0) for r in ok), 5),
        "duration": round(wall, 1),
        "lag_p95": _r(pct(lag, 95), 0),
    }
    row["slo_pass"] = level_passes(row, slo)
    return row


def level_passes(row: dict, slo: dict) -> bool:
    """Um nível cumpre o SLO quando o p95 de TTFT e de ITL ficam dentro do alvo
    e a taxa de erro não passa do limite."""
    if row["requests"] == 0 or row["ttft_p95"] is None:
        return False
    if row["err_pct"] > slo["max_err_pct"]:
        return False
    if row["ttft_p95"] > slo["ttft_ms"]:
        return False
    return row["itl_p95"] is None or row["itl_p95"] <= slo["itl_ms"]


def summarize_run(levels: list, slo: dict) -> list:
    return [summarize(lv["load"], lv["records"], lv["wall"], slo) for lv in levels]


def capacity(rows: list) -> dict:
    """Maior carga testada dentro do SLO e a primeira acima dela que falhou.

    Exige que todos os níveis abaixo também passem: um nível isolado que passa
    depois de uma falha (ruído) não conta como capacidade."""
    rows = sorted(rows, key=lambda r: r["load"])
    max_ok, first_fail = None, None
    for r in rows:
        if r["slo_pass"] and first_fail is None:
            max_ok = r["load"]
        elif not r["slo_pass"] and first_fail is None:
            first_fail = r["load"]
    best = next((r for r in rows if r["load"] == max_ok), None)
    return {
        "max_ok": max_ok,
        "first_fail": first_fail,
        "all_pass": first_fail is None and max_ok is not None,
        "goodput_rps": best["goodput_rps"] if best else None,
        "throughput": best["throughput"] if best else None,
    }


def run_closed_level(cfg: dict, level: int, reqs: list, on_progress) -> tuple:
    """Dispara len(reqs) requisições com `level` workers em paralelo."""
    records, t0 = [], time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=max(int(level), 1)) as ex:
        futs = [ex.submit(run_one, cfg, r) for r in reqs]
        for i, f in enumerate(cf.as_completed(futs), 1):
            records.append(f.result())
            on_progress(i, len(reqs))
    return records, time.perf_counter() - t0


def arrival_offsets(n: int, rate: float, rng: random.Random) -> list:
    """Instantes de chegada (s) de um processo de Poisson com taxa `rate`."""
    t, out = 0.0, []
    for i in range(n):
        out.append(t)
        if i < n - 1:
            t += rng.expovariate(rate)
    return out


def run_open_level(cfg: dict, rate: float, reqs: list, on_progress,
                   rng: random.Random) -> tuple:
    """Envia requisições em instantes de Poisson, sem esperar as anteriores
    terminarem — como usuários reais. Se o servidor não acompanha, a fila
    cresce e o TTFT sobe; em closed-loop isso fica escondido."""
    offsets = arrival_offsets(len(reqs), rate, rng)
    records = []
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=MAX_INFLIGHT) as ex:
        futs = []
        for at, req in zip(offsets, reqs):
            delay = t0 + at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            futs.append(ex.submit(run_one, cfg, req, t0 + at))
            on_progress(sum(f.done() for f in futs), len(reqs))
        for i, f in enumerate(cf.as_completed(futs), 1):
            records.append(f.result())
            on_progress(i, len(reqs))
    return records, time.perf_counter() - t0


def level_size(plan: dict, load: float) -> int:
    if plan["mode"] == "open":
        return max(1, round(load * plan["duration_s"]))
    return int(load) * int(plan["rounds"])


def plan_total(plan: dict) -> int:
    return sum(level_size(plan, lv) for lv in plan["levels"])


def warmup(cfg: dict, spec: dict, n: int = 2) -> list:
    """Requisições curtas descartadas: carregam kernels/caches e evitam que o
    primeiro nível pague o custo de "esquentar" o endpoint."""
    rng = random.Random()  # sem semente: prompts repetidos entre execuções gerariam prefix cache
    reqs = make_requests(n, {**spec, "in": min(spec["in"], 64), "out": 16,
                             "dist": "Fixo", "dataset": None}, rng)
    return [run_one(cfg, r) for r in reqs]


def run_sweep(cfg: dict, plan: dict, on_progress) -> dict:
    """Executa todos os níveis do plano. on_progress(done, total, load) roda
    na thread principal (pode atualizar a UI)."""
    rng = random.Random(plan.get("seed", 42))
    total = plan_total(plan)
    result = {"levels": [], "warmup": [], "started": time.time()}
    if plan.get("warmup"):
        result["warmup"] = warmup(cfg, plan["spec"])
    done = 0
    for load in sorted(plan["levels"]):
        n = level_size(plan, load)
        reqs = make_requests(n, plan["spec"], rng)

        def prog(i, _n, load=load, base=done):
            on_progress(base + i, total, load)

        t0 = time.time()
        if plan["mode"] == "open":
            records, wall = run_open_level(cfg, float(load), reqs, prog, rng)
        else:
            records, wall = run_closed_level(cfg, int(load), reqs, prog)
        result["levels"].append({"load": load, "wall": wall, "t0": t0,
                                 "t1": t0 + wall, "records": records})
        done += n
    result["ended"] = time.time()
    return result

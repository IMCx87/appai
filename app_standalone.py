"""
AI Infra Benchmark — versão ARQUIVO ÚNICO (gerada automaticamente)

Chat + benchmark para endpoints OpenAI-compatible (vLLM / Nutanix Enterprise AI)
com SLO, telemetria, cold start, RAG, histórico/A-B e custo — o mesmo app do
repositório, em um só arquivo para copiar e rodar onde quiser.

    pip install streamlit requests plotly
    streamlit run app_standalone.py

Mock vLLM/NAI sem GPU: botão na sidebar, ou
    python app_standalone.py mock --port 8000

Histórico: ./data/appai.db (mude com APPAI_DATA_DIR).
Tema escuro (opcional): streamlit run app_standalone.py --theme.base dark
(ou o .streamlit/config.toml do repositório ao lado do arquivo).

NÃO EDITE: gerado por tools/build_standalone.py a partir de appai/ e app.py
(fonte ad1049b4c5). Altere o pacote e rode o gerador de novo.
"""
# ruff: noqa
# fmt: off

import importlib.abc
import importlib.util
import linecache
import sys

_SOURCES = {
    'appai': r'''"""AI Infra Benchmark — lógica do app (sem dependência de Streamlit fora de appai.ui)."""
''',
    'appai.bench': r'''"""Motor de benchmark: closed-loop (concorrência fixa) e open-loop (Poisson),
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
''',
    'appai.charts': r'''"""Figuras Plotly compartilhadas entre a UI e o relatório HTML."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PURPLE, AMBER, TEAL, RED, BLUE, GRAY = "#7c6ff7", "#f59e0b", "#2dd4bf", "#f87171", "#60a5fa", "#9ca3af"
SERIES = [PURPLE, AMBER, TEAL, BLUE, RED, "#a3e635"]

COLUMNS = {
    "requests": "Requisições", "errors": "Erros", "err_pct": "Erros (%)",
    "retries": "Retries", "ttft_p50": "TTFT p50 (ms)", "ttft_p95": "TTFT p95 (ms)",
    "ttft_p99": "TTFT p99 (ms)", "itl_mean": "ITL médio (ms)", "itl_p95": "ITL p95 (ms)",
    "tps_mean": "tok/s por req", "e2e_p50": "E2E p50 (ms)", "e2e_p95": "E2E p95 (ms)",
    "throughput": "Vazão agregada (tok/s)", "rps": "Req/s", "goodput_rps": "Goodput (req/s)",
    "slo_ok_pct": "Dentro do SLO (%)", "slo_pass": "Nível no SLO", "tokens_in": "Tokens in",
    "tokens_out": "Tokens out", "cost": "Custo cloud (US$)", "duration": "Duração (s)",
    "lag_p95": "Atraso do gerador p95 (ms)",
}


def load_label(mode: str) -> str:
    return "Taxa de chegada (req/s)" if mode == "open" else "Concorrência"


def load_unit(mode: str) -> str:
    return "req/s" if mode == "open" else "usuários simultâneos"


def rows_df(rows: list, mode: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["slo_pass"] = df["slo_pass"].map({True: "✅", False: "❌"})
    return df.rename(columns={"load": load_label(mode), **COLUMNS})


def _layout(fig, title, height=360, **kw):
    fig.update_layout(title=title, height=height, margin=dict(t=50, b=10, l=60, r=60),
                      legend=dict(orientation="h", y=-0.25), **kw)
    return fig


def throughput_latency(rows: list, mode: str, slo: dict | None = None) -> go.Figure:
    x = [str(r["load"]) for r in rows]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = [TEAL if r["slo_pass"] else RED for r in rows]
    fig.add_trace(go.Scatter(x=x, y=[r["throughput"] for r in rows], mode="lines+markers",
                             name="Vazão agregada (tok/s)", line=dict(color=PURPLE, width=3),
                             marker=dict(color=colors, size=13,
                                         line=dict(color=PURPLE, width=2)),
                             customdata=[[r["goodput_rps"], r["slo_ok_pct"]] for r in rows],
                             hovertemplate="%{y:.0f} tok/s · goodput %{customdata[0]} req/s"
                                           " · %{customdata[1]}% no SLO<extra></extra>"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=[r["ttft_p95"] for r in rows], mode="lines+markers",
                             name="TTFT p95 (ms)",
                             line=dict(color=AMBER, width=3, dash="dot")), secondary_y=True)
    if slo:
        fig.add_trace(go.Scatter(x=x, y=[slo["ttft_ms"]] * len(x), mode="lines",
                                 name=f"SLO TTFT ({slo['ttft_ms']:.0f} ms)",
                                 line=dict(color=RED, width=1, dash="dash")), secondary_y=True)
    _layout(fig, "Vazão × latência por nível (marcador verde = nível dentro do SLO)")
    fig.update_xaxes(type="category", title_text=load_label(mode))
    fig.update_yaxes(title_text="tok/s", rangemode="tozero", secondary_y=False)
    fig.update_yaxes(title_text="ms", rangemode="tozero", showgrid=False, secondary_y=True)
    return fig


def box_by_level(levels: list, key: str, title: str, color: str, scale: float = 1.0,
                 mode: str = "closed") -> go.Figure:
    fig = go.Figure()
    for lv in levels:
        ys = [r[key] * scale for r in lv["records"] if r.get("ok") and r.get(key) is not None]
        fig.add_trace(go.Box(y=ys, name=str(lv["load"]), marker_color=color, boxmean=True))
    fig.update_xaxes(type="category", title_text=load_label(mode))
    return _layout(fig, title, height=320, showlegend=False)


def infra_timeline(levels: list, vllm_samples: list, prom_series: dict) -> go.Figure | None:
    """Telemetria no tempo, com as janelas de cada nível sombreadas.
    Painel de cima: vLLM (fila/execução à esquerda, KV cache % à direita).
    Painel de baixo: séries do Prometheus (% à direita, demais à esquerda)."""
    has_prom = any(prom_series.values()) if prom_series else False
    if not vllm_samples and not has_prom:
        return None
    t_ref = levels[0]["t0"] if levels else 0
    nrows = int(bool(vllm_samples)) + int(has_prom)
    titles = ([("vLLM: fila, execução e KV cache")] if vllm_samples else []) + \
             (["GPU / Prometheus"] if has_prom else [])
    fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=titles,
                        specs=[[{"secondary_y": True}] for _ in range(nrows)])
    row = 1
    if vllm_samples:
        ts = [s["t"] - t_ref for s in vllm_samples]
        for key, name, color in (("waiting", "Fila (waiting)", RED),
                                 ("running", "Em execução (running)", BLUE)):
            ys = [s.get(key) for s in vllm_samples]
            if any(v is not None for v in ys):
                fig.add_trace(go.Scatter(x=ts, y=ys, name=name, mode="lines",
                                         line=dict(color=color, width=2, shape="hv")),
                              row=row, col=1, secondary_y=False)
        kv = [None if s.get("kv") is None else s["kv"] * 100 for s in vllm_samples]
        if any(v is not None for v in kv):
            fig.add_trace(go.Scatter(x=ts, y=kv, name="KV cache (%)", mode="lines",
                                     line=dict(color=PURPLE, width=3)),
                          row=row, col=1, secondary_y=True)
        fig.update_yaxes(title_text="requisições", rangemode="tozero", row=row, col=1,
                         secondary_y=False)
        fig.update_yaxes(title_text="KV %", range=[0, 105], row=row, col=1, secondary_y=True)
        row += 1
    if has_prom:
        palette = iter([AMBER, TEAL, GRAY, "#a3e635", BLUE, RED])
        for name, pts in prom_series.items():
            if not pts:
                continue
            fig.add_trace(go.Scatter(x=[t - t_ref for t, _ in pts], y=[v for _, v in pts],
                                     name=name, mode="lines",
                                     line=dict(color=next(palette, GRAY), width=2)),
                          row=row, col=1, secondary_y="%" in name)
        fig.update_yaxes(title_text="valor", rangemode="tozero", row=row, col=1,
                         secondary_y=False)
        fig.update_yaxes(title_text="%", range=[0, 105], row=row, col=1, secondary_y=True)
    for i, lv in enumerate(levels):
        fig.add_vrect(x0=lv["t0"] - t_ref, x1=lv["t1"] - t_ref,
                      fillcolor=PURPLE if i % 2 == 0 else TEAL, opacity=0.08, line_width=0,
                      row="all", col=1)
        fig.add_annotation(x=(lv["t0"] + lv["t1"]) / 2 - t_ref, y=1.0, yref="paper",
                           text=f"<b>{lv['load']:g}</b>", showarrow=False, yshift=14,
                           font=dict(size=11))
    _layout(fig, "Telemetria do servidor durante o benchmark (faixas = níveis de carga)",
            height=260 + 200 * nrows)
    fig.update_layout(margin=dict(t=80, b=10, l=60, r=60))
    fig.update_xaxes(title_text="segundos desde o início", row=nrows, col=1)
    return fig


def compare_runs(runs: list, key: str, title: str) -> go.Figure:
    """runs: [{name, mode, rows}] — uma linha por execução."""
    fig = go.Figure()
    for i, r in enumerate(runs):
        rows = sorted(r["rows"], key=lambda x: x["load"])
        fig.add_trace(go.Scatter(x=[x["load"] for x in rows], y=[x.get(key) for x in rows],
                                 mode="lines+markers", name=r["name"],
                                 line=dict(color=SERIES[i % len(SERIES)], width=3)))
    mode = runs[0]["mode"] if runs else "closed"
    loads = sorted({x["load"] for r in runs for x in r["rows"]})
    fig.update_xaxes(title_text=load_label(mode), type="log" if _wide(runs) else "linear",
                     tickvals=loads, ticktext=[f"{v:g}" for v in loads])
    return _layout(fig, title, height=330)


def _wide(runs):
    loads = [x["load"] for r in runs for x in r["rows"] if x["load"]]
    return bool(loads) and max(loads) / min(loads) >= 16


def coldstart_bars(runs: list) -> go.Figure:
    # nomes repetidos virariam uma barra só (o Plotly empilha mesma categoria)
    seen: dict = {}
    names = []
    for r in runs:
        seen[r["name"]] = seen.get(r["name"], 0) + 1
        names.append(r["name"] if seen[r["name"]] == 1 else f"{r['name']} ({seen[r['name']]})")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[r.get("ready_s") for r in runs], name="Até responder (s)",
                         marker_color=PURPLE))
    fig.add_trace(go.Bar(x=names, y=[(r.get("total_s") or 0) - (r.get("ready_s") or 0)
                                     for r in runs],
                         name="Até o 1º token, após responder (s)", marker_color=AMBER))
    fig.update_layout(barmode="stack")
    return _layout(fig, "Cold start por execução", height=340)


def rag_stages(summary: dict) -> go.Figure:
    stages = [s for s in summary["stages"] if s["key"] in
              ("embed_ms", "retrieve_ms", "prefill_ms", "decode_ms")]
    fig = go.Figure()
    for s, color in zip(stages, [BLUE, TEAL, AMBER, PURPLE]):
        fig.add_trace(go.Bar(y=["p50", "p95"], x=[s["p50 (ms)"], s["p95 (ms)"]],
                             name=s["Etapa"], orientation="h", marker_color=color))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="ms")
    return _layout(fig, "Onde está o tempo de uma resposta RAG", height=300)


def cost_curve_fig(curve: list, volume: float) -> go.Figure:
    x = [c["volume_out_m"] for c in curve]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[c["cloud"] for c in curve], name="Cloud (por token)",
                             line=dict(color=AMBER, width=3)))
    fig.add_trace(go.Scatter(x=x, y=[c["onprem"] for c in curve], name="On-prem (GPU própria)",
                             line=dict(color=PURPLE, width=3, shape="hv")))
    if volume:
        fig.add_vline(x=volume, line_dash="dash", line_color=GRAY,
                      annotation_text="volume informado")
    fig.update_xaxes(title_text="Milhões de tokens de saída por mês")
    fig.update_yaxes(title_text="US$ / mês")
    return _layout(fig, "Custo mensal × volume (degraus = mais um conjunto de GPUs)", height=360)
''',
    'appai.client': r'''"""Cliente HTTP para endpoints OpenAI-compatible (vLLM / Nutanix Enterprise AI).

Nada aqui usa st.session_state: as funções recebem um `cfg` (dict) e podem
rodar em threads de benchmark.

cfg = {base, headers, verify, model, temperature, max_retries, prices, extra}
"""

import json
import random
import time

import requests

from appai.cost import estimate_cost

AUTH_TYPES = ["Authorization: Bearer {key}", "x-api-key: {key}", "None / No Auth"]


def api_base(url: str) -> str:
    base = (url or "").strip().rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def build_headers(key: str, auth_type: str) -> dict:
    h = {"Content-Type": "application/json"}
    key = (key or "").strip()
    if auth_type.startswith("Authorization"):
        h["Authorization"] = f"Bearer {key}"
    elif auth_type.startswith("x-api-key"):
        h["x-api-key"] = key
    return h


def needs_key(auth_type: str) -> bool:
    return not auth_type.startswith("None")


def redact(headers: dict) -> dict:
    return {
        k: (v[:14] + "…" if k in ("Authorization", "x-api-key") else v)
        for k, v in headers.items()
    }


def sanitize_messages(messages: list) -> list:
    """Garante alternância estrita user/assistant, exigida por alguns chat templates."""
    filtered = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant")
    ]
    if not filtered:
        return []
    merged = [dict(filtered[0])]
    for msg in filtered[1:]:
        if msg["role"] == merged[-1]["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(dict(msg))
    while merged and merged[0]["role"] != "user":
        merged.pop(0)
    return merged


def fetch_models(cfg: dict) -> list:
    r = requests.get(cfg["base"] + "/models", headers=cfg["headers"],
                     timeout=10, verify=cfg["verify"])
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


# ── Retry para erros transitórios ─────────────────────────────────────────────
# O LB do NAI pode rotear para um backend hibernado e devolver
# 404 {"message": "... hibernated endpoint"}. Como o balanceamento distribui
# entre réplicas, uma nova tentativa tende a cair em um backend saudável.
# Também cobre os transitórios clássicos (429/5xx).
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_HINTS = ("hibernated", "hibernating")


def is_retryable(status: int, body: str) -> bool:
    if status in RETRY_STATUS:
        return True
    low = (body or "").lower()
    return status == 404 and any(h in low for h in RETRY_HINTS)


def is_hibernated(status: int, body: str) -> bool:
    return status == 404 and "hibernat" in (body or "").lower()


def post_with_retry(cfg: dict, body: dict, stream: bool, notify=None, log=None,
                    path: str = "/chat/completions"):
    """POST com retry exponencial + jitter. Retorna (Response, nº de retries).
    notify(msg) é chamado a cada nova tentativa; log (lista) recebe o detalhe
    de cada falha."""
    endpoint = cfg["base"] + path
    max_retries = max(1, int(cfg.get("max_retries", 1)))
    retries = 0
    resp = None
    for attempt in range(1, max_retries + 1):
        resp = requests.post(
            endpoint, headers=cfg["headers"], json=body, stream=stream,
            timeout=(10, 300), verify=cfg["verify"],
        )
        if resp.status_code < 400:
            return resp, retries
        err_body = resp.text[:1500]
        if log is not None:
            log.append(f"Tentativa {attempt}/{max_retries}: HTTP {resp.status_code}\n{err_body}")
        if attempt < max_retries and is_retryable(resp.status_code, err_body):
            retries += 1
            wait = min(2 ** attempt, 10) + random.uniform(0, 0.5)
            if notify:
                hib = "endpoint hibernado no pool" if is_hibernated(resp.status_code, err_body) \
                      else f"HTTP {resp.status_code}"
                notify(f"⏳ {hib}. Nova tentativa {attempt + 1}/{max_retries} "
                       f"em {wait:.1f}s…")
            resp.close()
            time.sleep(wait)
            continue
        resp.raise_for_status()
    resp.raise_for_status()


def chat_body(cfg: dict, messages: list, max_tokens: int, stream: bool = True) -> dict:
    body = {
        "model": cfg["model"],
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": cfg.get("temperature", 0.7),
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    body.update(cfg.get("extra") or {})
    return body


def stream_response(resp, metrics: dict):
    """Generator SSE: entrega texto token a token e mede TTFT / throughput."""
    metrics.update(
        {"start": metrics.get("start", time.perf_counter()), "ttft": None,
         "chunks": 0, "usage": None, "model": None, "end": None,
         "t_first": None, "t_last": None}
    )
    with resp:
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("model"):
                metrics["model"] = chunk["model"]
            if chunk.get("usage"):
                metrics["usage"] = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices:
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    now = time.perf_counter()
                    if metrics["ttft"] is None:
                        metrics["ttft"] = now - metrics["start"]
                        metrics["t_first"] = now
                    metrics["t_last"] = now
                    metrics["chunks"] += 1
                    yield delta
    metrics["end"] = time.perf_counter()


def finalize_metrics(metrics: dict, model: str, prices: tuple) -> dict:
    meta = {"model": metrics.get("model") or model}
    usage = metrics.get("usage") or {}
    out_tokens = usage.get("completion_tokens") or metrics.get("chunks") or 0
    meta["in"] = usage.get("prompt_tokens", 0) or 0
    meta["out"] = out_tokens
    meta["usage_reported"] = bool(usage)
    ttft = metrics.get("ttft")
    end, start = metrics.get("end"), metrics.get("start")
    if ttft is not None:
        meta["ttft"] = ttft
    if end and start and ttft is not None and out_tokens > 1:
        gen_time = max(end - start - ttft, 1e-6)
        meta["tps"] = out_tokens / gen_time
    t_first, t_last = metrics.get("t_first"), metrics.get("t_last")
    chunks = metrics.get("chunks", 0)
    if t_first and t_last and chunks > 1:
        # ITL por chunk SSE; quando o servidor agrupa tokens por chunk, usa a
        # contagem real de tokens para não superestimar a latência entre tokens.
        steps = max(chunks, out_tokens) - 1
        meta["itl"] = (t_last - t_first) / steps * 1000  # ms entre tokens
    meta["cost"] = estimate_cost(meta["in"], meta["out"], prices)
    return meta


def call_blocking(cfg: dict, body: dict, notify=None, log=None) -> tuple:
    start = time.perf_counter()
    resp, _ = post_with_retry(cfg, body, stream=False, notify=notify, log=log)
    if log is not None:
        log.append(f"Status: {resp.status_code}\nResponse: {resp.text[:2000]}")
    data = resp.json()
    elapsed = time.perf_counter() - start
    usage = data.get("usage", {})
    content = data["choices"][0]["message"]["content"]
    out = usage.get("completion_tokens", 0)
    meta = {
        "model": data.get("model", cfg["model"]),
        "in": usage.get("prompt_tokens", 0),
        "out": out,
    }
    if out > 0 and elapsed > 0:
        meta["tps"] = out / elapsed
    meta["cost"] = estimate_cost(meta["in"], meta["out"], cfg["prices"])
    return content, meta


def stream_collect(cfg: dict, messages: list, max_tokens: int) -> tuple:
    """Faz uma chamada em streaming e devolve (texto, meta). Usado pelo modo
    lado a lado e pelo RAG, onde o texto é exibido só no final."""
    body = chat_body(cfg, messages, max_tokens, stream=True)
    metrics = {"start": time.perf_counter()}
    resp, retries = post_with_retry(cfg, body, stream=True)
    text = "".join(stream_response(resp, metrics))
    meta = finalize_metrics(metrics, cfg["model"], cfg["prices"])
    meta["retries"] = retries
    meta["latency"] = metrics["end"] - metrics["start"]
    return text, meta


def embed(cfg: dict, inputs: list, model: str) -> list:
    """POST /v1/embeddings. Devolve a lista de vetores na ordem de `inputs`."""
    resp, _ = post_with_retry(cfg, {"model": model, "input": inputs},
                              stream=False, path="/embeddings")
    data = resp.json().get("data", [])
    data = sorted(data, key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]
''',
    'appai.coldstart': r'''"""Cold start: quanto tempo um endpoint hibernado (ou escalado a zero) leva
até responder o primeiro token.

Boa parte desse tempo é carregar dezenas de GB de pesos do storage para a
GPU — por isso a medição é rotulada pelo backend de storage (Nutanix Files,
Volumes, disco local...). O tempo também inclui agendamento do pod, pull de
imagem e inicialização do runtime: a vazão de carga derivada é uma estimativa
por baixo, não a vazão do storage.
"""

import time

import requests

from appai.client import chat_body, finalize_metrics, is_retryable, stream_response

STORAGE_BACKENDS = ["Nutanix Files (NFS)", "Nutanix Volumes (iSCSI)",
                    "Nutanix Objects (S3)", "Disco local / NVMe", "Outro"]


def measure_cold_start(cfg: dict, prompt: str = "Responda apenas: OK.",
                       timeout_s: float = 900, poll_s: float = 5,
                       on_event=None, should_stop=None) -> dict:
    """Tenta até o endpoint responder. Registra cada tentativa.

    Retorna {ok, ready_s, ttft_s, total_s, attempts, events, meta, error}.
    ready_s = do primeiro envio até a resposta HTTP de sucesso;
    total_s = do primeiro envio até o primeiro token."""
    t0 = time.perf_counter()
    wall0 = time.time()
    events = []

    def ev(status, note):
        e = {"t": round(time.perf_counter() - t0, 2), "status": status, "note": note}
        events.append(e)
        if on_event:
            on_event(e)

    body = chat_body(cfg, [{"role": "user", "content": prompt}], 16, stream=True)
    attempts = 0
    while True:
        if should_stop and should_stop():
            return _result(False, t0, wall0, attempts, events, error="interrompido")
        elapsed = time.perf_counter() - t0
        if elapsed > timeout_s:
            return _result(False, t0, wall0, attempts, events,
                           error=f"timeout após {timeout_s:.0f}s")
        attempts += 1
        t_try = time.perf_counter()
        try:
            resp = requests.post(cfg["base"] + "/chat/completions", headers=cfg["headers"],
                                 json=body, stream=True, timeout=(10, 300),
                                 verify=cfg["verify"])
        except (requests.ConnectionError, requests.Timeout) as e:
            ev("conn", f"sem conexão ({type(e).__name__})")
            time.sleep(poll_s)
            continue
        if resp.status_code >= 400:
            txt = resp.text[:300]
            resp.close()
            if is_retryable(resp.status_code, txt):
                note = "hibernado/acordando" if "hibernat" in txt.lower() else txt[:80]
                ev(resp.status_code, note)
                time.sleep(poll_s)
                continue
            ev(resp.status_code, txt[:120])
            return _result(False, t0, wall0, attempts, events,
                           error=f"HTTP {resp.status_code}: {txt[:200]}")
        ready = time.perf_counter() - t0
        ev(resp.status_code, "respondeu")
        metrics = {"start": t_try}
        for _ in stream_response(resp, metrics):
            pass
        meta = finalize_metrics(metrics, cfg["model"], cfg["prices"])
        total = (metrics["t_first"] - t0) if metrics.get("t_first") else None
        return _result(True, t0, wall0, attempts, events, ready=ready, total=total,
                       meta=meta)


def _result(ok, t0, wall0, attempts, events, ready=None, total=None, meta=None, error=""):
    return {
        "ok": ok,
        "ready_s": ready,
        "total_s": total,
        "ttft_s": (meta or {}).get("ttft"),
        "attempts": attempts,
        "events": events,
        "meta": meta or {},
        "error": error,
        "started": wall0,
        "elapsed_s": time.perf_counter() - t0,
    }


def load_throughput_gbs(model_gb: float, seconds: float | None) -> float | None:
    """GB/s efetivos (pesos ÷ tempo até pronto). Limite inferior: o tempo
    inclui tudo que não é leitura de storage."""
    if not model_gb or not seconds:
        return None
    return model_gb / seconds
''',
    'appai.cost': r'''"""Modelo de custo: cloud pública (por token) × on-prem (GPU própria).

Tudo aqui são premissas editáveis, não medições — exceto a vazão (tok/s) e,
quando há telemetria DCGM, a potência medida durante o benchmark.
"""

import math

HOURS_PER_MONTH = 730

# Preços ILUSTRATIVOS por 1M tokens (entrada, saída) em US$ — ajuste ao provedor.
CLOUD_PRESETS = {
    "Modelo pequeno (~8B)": (0.15, 0.60),
    "Modelo médio (~70B)": (0.90, 0.90),
    "Modelo frontier": (3.00, 15.00),
    "Personalizado": None,
}


def estimate_cost(in_tokens: int, out_tokens: int, prices: tuple) -> float:
    """Custo estimado se a mesma troca rodasse em uma cloud pública
    (preços em US$ por 1M de tokens: entrada, saída)."""
    return (in_tokens or 0) / 1e6 * prices[0] + (out_tokens or 0) / 1e6 * prices[1]


def fmt_cost(cost: float) -> str:
    return f"US$ {cost:.5f}" if cost < 0.01 else f"US$ {cost:.4f}"


def energy_month_usd(power_w: float, gpu_count: int, pue: float, kwh_price: float) -> float:
    """Energia de um conjunto de GPUs por mês (potência média × PUE)."""
    return power_w * gpu_count * pue * HOURS_PER_MONTH / 1000 * kwh_price


def onprem_analysis(p: dict) -> dict:
    """Compara on-prem × cloud para um volume mensal.

    p: peak_tps (tok/s de saída medidos), utilization (0–1), gpu_hour, gpu_count,
       monthly_in_m, monthly_out_m (milhões de tokens), price_in, price_out,
       energy_separate (bool), power_w, pue, kwh_price.

    A vazão medida já embute a proporção entrada/saída do benchmark; a
    capacidade é expressa em tokens de saída, processando a entrada junto.
    """
    peak = max(float(p.get("peak_tps") or 0), 0.0)
    util = min(max(float(p.get("utilization", 1.0)), 0.01), 1.0)
    gpu_count = max(int(p.get("gpu_count", 1)), 1)
    vol_in = max(float(p.get("monthly_in_m", 0)), 0.0)
    vol_out = max(float(p.get("monthly_out_m", 0)), 0.0)
    price_in, price_out = float(p.get("price_in", 0)), float(p.get("price_out", 0))

    hw_month = float(p.get("gpu_hour", 0)) * gpu_count * HOURS_PER_MONTH
    energy_month = 0.0
    if p.get("energy_separate"):
        energy_month = energy_month_usd(float(p.get("power_w", 0)), gpu_count,
                                        float(p.get("pue", 1.0)), float(p.get("kwh_price", 0)))
    set_month = hw_month + energy_month  # custo mensal de UM endpoint (conjunto de GPUs)

    capacity_out_m = peak * 3600 * HOURS_PER_MONTH * util / 1e6
    replicas = math.ceil(vol_out / capacity_out_m) if capacity_out_m > 0 and vol_out > 0 else 1
    replicas = max(replicas, 1)
    onprem_month = set_month * replicas

    ratio = vol_in / vol_out if vol_out > 0 else 0.0
    cloud_per_m_out = price_out + ratio * price_in  # custo cloud por 1M de saída (com a entrada)
    cloud_month = vol_in * price_in + vol_out * price_out

    onprem_per_m_out = set_month / capacity_out_m if capacity_out_m > 0 else None
    breakeven_m = set_month / cloud_per_m_out if cloud_per_m_out > 0 else None

    return {
        "hw_month": hw_month,
        "energy_month": energy_month,
        "set_month": set_month,
        "capacity_out_m": capacity_out_m,
        "replicas": replicas,
        "onprem_month": onprem_month,
        "cloud_month": cloud_month,
        "cloud_per_m_out": cloud_per_m_out,
        "onprem_per_m_out": onprem_per_m_out,
        "breakeven_m": breakeven_m,
        "breakeven_within_capacity": (breakeven_m is not None
                                      and breakeven_m <= capacity_out_m),
        "in_out_ratio": ratio,
        "savings_month": cloud_month - onprem_month,
    }


def cost_curve(p: dict, points: int = 40) -> list:
    """Custo mensal cloud × on-prem para volumes de 0 a ~3× o informado (ou a
    capacidade), para o gráfico de break-even. On-prem é uma escada: cada vez
    que o volume passa da capacidade, entra mais um conjunto de GPUs."""
    base = onprem_analysis(p)
    cap = base["capacity_out_m"]
    top = max(float(p.get("monthly_out_m", 0)) * 3, cap * 2.5, 1.0)
    ratio = base["in_out_ratio"]
    out = []
    for i in range(points + 1):
        vol = top * i / points
        reps = max(math.ceil(vol / cap), 1) if cap > 0 else 1
        out.append({
            "volume_out_m": vol,
            "cloud": vol * float(p.get("price_out", 0)) + vol * ratio * float(p.get("price_in", 0)),
            "onprem": base["set_month"] * reps,
        })
    return out


def energy_per_m_tokens(energy_kwh: float, out_tokens: int) -> float | None:
    """kWh por 1M de tokens de saída, a partir da energia medida (DCGM)."""
    if not out_tokens or energy_kwh is None:
        return None
    return energy_kwh / out_tokens * 1e6
''',
    'appai.mock_server': r'''"""Servidor mock OpenAI-compatible que imita um vLLM/NAI — para desenvolver,
testar e ensaiar demos sem GPU.

    python -m appai.mock_server --port 8000
    # no app: API Endpoint URL = http://localhost:8000/v1, auth = None
    # vLLM metrics URL = http://localhost:8000/metrics
    # Prometheus URL   = http://localhost:8000      (DCGM sintético)

Simula: streaming SSE com usage, continuous batching com limite de batch
(fila quando estoura), KV cache finito com preempções, prefix cache,
hibernação (404 "hibernated endpoint" até acordar), /v1/embeddings
determinístico (bag-of-words com hash) e uma API Prometheus mínima com
métricas DCGM sintéticas. Os números são plausíveis, não reais.
"""

import argparse
import hashlib
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

WORD = re.compile(r"\w+", re.UNICODE)
EMB_DIM = 256
LOREM = ("A inferência de modelos de linguagem depende de GPU, memória e storage. "
         "O KV cache cresce com o contexto e com o número de usuários simultâneos. ").split()


class Sim:
    def __init__(self, decode_tps=45.0, prefill_tps=6000.0, max_batch=16,
                 kv_tokens=40000, wake_s=0.0, hibernated=False, natural_out=120,
                 model="mock-llm", emb_model="mock-embed"):
        self.decode_tps, self.prefill_tps = decode_tps, prefill_tps
        self.max_batch, self.kv_tokens = max_batch, kv_tokens
        self.wake_s, self.natural_out = wake_s, natural_out
        self.model, self.emb_model = model, emb_model
        self.lock = threading.Lock()
        self.slots = threading.Semaphore(max_batch)
        self.running = self.waiting = self.kv_used = 0
        self.preempt = self.prefix_hits = self.prefix_queries = 0
        self.prefixes: set = set()
        self.hibernated, self.wake_at = hibernated, None
        self.history: list = []
        self._stop = threading.Event()
        threading.Thread(target=self._sample, daemon=True).start()

    # métricas "DCGM" derivadas do estado
    def gpu_state(self):
        load = min(self.running / max(self.max_batch * 0.5, 1), 1.0)
        util = 0.0 if self.hibernated else 6 + 90 * load
        kv = min(self.kv_used / self.kv_tokens, 1.0)
        mem = 0.0 if self.hibernated else 18000 + 22000 * kv
        power = 25.0 if self.hibernated else 70 + util * 3.2
        return util, mem, power, kv

    def _sample(self):
        while not self._stop.is_set():
            util, mem, power, _ = self.gpu_state()
            self.history.append((time.time(), util, mem, power))
            self.history = self.history[-20000:]
            self._stop.wait(0.5)

    def hibernate(self):
        with self.lock:
            self.hibernated, self.wake_at = True, None

    def check_awake(self) -> bool:
        with self.lock:
            if not self.hibernated:
                return True
            now = time.time()
            if self.wake_at is None:
                self.wake_at = now + self.wake_s
            if now >= self.wake_at:
                self.hibernated, self.wake_at = False, None
                return True
            return False


def approx_tokens(text: str) -> int:
    return max(1, round(len(text.split()) / 0.75))


def embed_text(text: str) -> list:
    v = [0.0] * EMB_DIM
    for w in WORD.findall(text.lower()):
        if len(w) < 3:
            continue
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % EMB_DIM] += 1.0
    return v


def make_handler(sim: Sim):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):  # silencioso
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/v1/models", "/models"):
                return self._json(200, {"object": "list", "data": [
                    {"id": sim.model, "object": "model"},
                    {"id": sim.emb_model, "object": "model"}]})
            if u.path == "/metrics":
                return self._metrics()
            if u.path == "/api/v1/query_range":
                return self._query_range(parse_qs(u.query))
            if u.path in ("/health", "/"):
                return self._json(200, {"status": "ok", "hibernated": sim.hibernated})
            self._json(404, {"message": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            if u.path == "/admin/hibernate":
                sim.hibernate()
                return self._json(200, {"hibernated": True})
            if u.path in ("/v1/embeddings", "/embeddings"):
                body = self._body()
                inputs = body.get("input")
                inputs = [inputs] if isinstance(inputs, str) else inputs or []
                time.sleep(0.004 + 0.0008 * len(inputs))
                return self._json(200, {"object": "list", "model": body.get("model"),
                                        "data": [{"object": "embedding", "index": i,
                                                  "embedding": embed_text(t)}
                                                 for i, t in enumerate(inputs)],
                                        "usage": {"prompt_tokens": sum(
                                            approx_tokens(t) for t in inputs)}})
            if u.path in ("/v1/chat/completions", "/chat/completions"):
                return self._chat(self._body())
            self._json(404, {"message": "not found"})

        def _chat(self, body):
            if not sim.check_awake():
                return self._json(404, {"message": "Request routed to hibernated endpoint"})
            prompt = "\n".join(str(m.get("content", "")) for m in body.get("messages", []))
            tin = approx_tokens(prompt)
            max_tokens = int(body.get("max_tokens") or 256)
            tout = max_tokens if body.get("ignore_eos") else min(max_tokens, sim.natural_out)
            stream = bool(body.get("stream"))
            with sim.lock:
                sim.waiting += 1
            sim.slots.acquire()
            try:
                with sim.lock:
                    sim.waiting -= 1
                    sim.running += 1
                    sim.kv_used += tin + tout
                    pressure = sim.kv_used > sim.kv_tokens
                    if pressure:
                        sim.preempt += 1
                    key = prompt[:64]
                    sim.prefix_queries += tin
                    cached = key in sim.prefixes
                    if cached:
                        sim.prefix_hits += int(tin * 0.8)
                    sim.prefixes.add(key)
                    running = sim.running
                eff_in = tin * (0.2 if cached else 1.0)
                prefill = 0.03 + eff_in / sim.prefill_tps * (1 + 0.15 * (running - 1))
                if pressure:
                    prefill += 0.4
                per_tok = 1.0 / sim.decode_tps * (1 + 0.07 * (running - 1))
                rid = f"chatcmpl-{int(time.time() * 1000)}"
                if not stream:
                    time.sleep(prefill + per_tok * tout)
                    return self._json(200, {
                        "id": rid, "object": "chat.completion", "model": sim.model,
                        "choices": [{"index": 0, "finish_reason": "stop", "message": {
                            "role": "assistant",
                            "content": " ".join(LOREM[i % len(LOREM)] for i in range(tout))}}],
                        "usage": {"prompt_tokens": tin, "completion_tokens": tout,
                                  "total_tokens": tin + tout}})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                time.sleep(prefill)
                for i in range(tout):
                    chunk = {"id": rid, "object": "chat.completion.chunk", "model": sim.model,
                             "choices": [{"index": 0, "delta": {
                                 "content": LOREM[i % len(LOREM)] + " "}}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(per_tok)
                if (body.get("stream_options") or {}).get("include_usage"):
                    usage = {"id": rid, "object": "chat.completion.chunk", "model": sim.model,
                             "choices": [], "usage": {"prompt_tokens": tin,
                                                      "completion_tokens": tout,
                                                      "total_tokens": tin + tout}}
                    self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with sim.lock:
                    sim.running -= 1
                    sim.kv_used -= tin + tout
                sim.slots.release()

        def _metrics(self):
            _, _, _, kv = sim.gpu_state()
            m = f'model_name="{sim.model}"'
            text = (
                "# HELP vllm:num_requests_running Requests running.\n"
                "# TYPE vllm:num_requests_running gauge\n"
                f"vllm:num_requests_running{{{m}}} {sim.running}\n"
                "# TYPE vllm:num_requests_waiting gauge\n"
                f"vllm:num_requests_waiting{{{m}}} {sim.waiting}\n"
                "# TYPE vllm:kv_cache_usage_perc gauge\n"
                f"vllm:kv_cache_usage_perc{{{m}}} {kv:.4f}\n"
                "# TYPE vllm:num_preemptions_total counter\n"
                f"vllm:num_preemptions_total{{{m}}} {sim.preempt}\n"
                "# TYPE vllm:prefix_cache_queries_total counter\n"
                f"vllm:prefix_cache_queries_total{{{m}}} {sim.prefix_queries}\n"
                "# TYPE vllm:prefix_cache_hits_total counter\n"
                f"vllm:prefix_cache_hits_total{{{m}}} {sim.prefix_hits}\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(text)))
            self.end_headers()
            self.wfile.write(text)

        def _query_range(self, qs):
            q = (qs.get("query") or [""])[0].upper()
            start = float((qs.get("start") or ["0"])[0])
            end = float((qs.get("end") or [str(time.time())])[0])
            step = float((qs.get("step") or ["1s"])[0].rstrip("s") or 1)
            col = 1 if "UTIL" in q else 2 if "FB_USED" in q else 3 if "POWER" in q else None
            if col is None:
                return self._json(200, {"status": "success",
                                        "data": {"resultType": "matrix", "result": []}})
            div = 1024.0 if col == 2 and "1024" in q else 1.0
            values, next_t = [], start
            for row in sim.history:
                if start <= row[0] <= end and row[0] >= next_t:
                    values.append([row[0], f"{row[col] / div:.3f}"])
                    next_t = row[0] + step
            return self._json(200, {"status": "success", "data": {
                "resultType": "matrix", "result": [{"metric": {"gpu": "0"}, "values": values}]}})

    return H


class _Server(ThreadingHTTPServer):
    # O padrão do socketserver é 5 conexões pendentes: com dezenas de usuários
    # simultâneos o kernel descarta SYNs e o cliente só tenta de novo após ~1 s
    # (retransmissão TCP) — um TTFT falso que nada tem a ver com a simulação.
    request_queue_size = 1024
    daemon_threads = True


def serve(port: int = 8000, host: str = "127.0.0.1", **kw):
    """Sobe o mock em uma thread; devolve (server, sim)."""
    sim = Sim(**kw)
    srv = _Server((host, port), make_handler(sim))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, sim


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--decode-tps", type=float, default=45.0, help="tok/s de uma requisição sozinha")
    ap.add_argument("--max-batch", type=int, default=16, help="requisições simultâneas antes de enfileirar")
    ap.add_argument("--kv-tokens", type=int, default=40000, help="capacidade do KV cache em tokens")
    ap.add_argument("--wake-seconds", type=float, default=20.0, help="tempo para acordar da hibernação")
    ap.add_argument("--hibernated", action="store_true", help="inicia hibernado")
    a = ap.parse_args()
    srv, _ = serve(a.port, a.host, decode_tps=a.decode_tps, max_batch=a.max_batch,
                   kv_tokens=a.kv_tokens, wake_s=a.wake_seconds, hibernated=a.hibernated)
    print(f"mock vLLM/NAI em http://{a.host}:{a.port}/v1  (metrics: /metrics, "
          f"prometheus: /api/v1/query_range, hibernar: POST /admin/hibernate)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
''',
    'appai.rag': r'''"""RAG medido por etapa: embedding da pergunta → busca vetorial → geração.

O "TTFT que o usuário sente" num assistente RAG é embedding + busca + TTFT do
modelo. Medir cada etapa separadamente mostra onde está o tempo — e o prompt
com contexto recuperado é bem maior que um chat, o que pesa no prefill.

A busca vetorial aqui é em memória (numpy, força bruta por cosseno): mede o
pipeline e o endpoint de embeddings, não um banco vetorial de produção.
"""

import concurrent.futures as cf
import random
import statistics
import time

import numpy as np

from appai.bench import pct
from appai.client import chat_body, embed, finalize_metrics, post_with_retry, stream_response

SAMPLE_CORPUS = [
    ("KV cache", "Durante a inferência de um LLM, o KV cache guarda as chaves e valores de "
     "atenção de cada token já processado. Ele cresce com o tamanho do contexto e com o "
     "número de requisições simultâneas, e vive na memória da GPU. Quando o KV cache "
     "enche, servidores como o vLLM passam a enfileirar requisições ou a preemptar as "
     "que estão em andamento, o que aumenta a latência de cauda. Quantizar o KV cache "
     "em FP8 ou reduzir o tamanho máximo de contexto libera memória para mais usuários."),
    ("Prefill e decode", "A geração de texto tem duas fases. No prefill o modelo processa "
     "o prompt inteiro de uma vez; é uma fase limitada por computação e determina o tempo "
     "até o primeiro token (TTFT). No decode o modelo gera um token por vez; é uma fase "
     "limitada por largura de banda de memória e determina a latência entre tokens (ITL). "
     "Prompts longos, como os de RAG, aumentam o custo de prefill."),
    ("Continuous batching", "O continuous batching permite que o servidor de inferência "
     "adicione novas requisições ao lote a cada passo de decode, em vez de esperar o lote "
     "inteiro terminar. Isso aumenta muito a vazão agregada, mas cada requisição divide a "
     "GPU com as demais: com mais usuários simultâneos a vazão total sobe até saturar, "
     "enquanto a velocidade individual de cada resposta cai."),
    ("Prefix caching", "O prefix caching reaproveita o KV cache de prompts que começam com "
     "o mesmo texto, como um system prompt longo ou o mesmo documento consultado várias "
     "vezes. O prefill desse trecho deixa de ser recalculado, reduzindo o TTFT e o uso de "
     "GPU. Benchmarks que repetem o mesmo prompt podem parecer melhores do que a carga "
     "real por causa desse efeito."),
    ("Storage para modelos", "Os pesos de um modelo de linguagem ocupam de alguns GB a "
     "centenas de GB. Ao iniciar ou acordar um endpoint, esses pesos são lidos do storage "
     "e carregados na GPU. A vazão de leitura sequencial do storage — NFS, iSCSI, objeto "
     "ou NVMe local — influencia diretamente o tempo de cold start e a agilidade para "
     "escalar réplicas sob demanda."),
    ("Snapshots e proteção de dados", "Snapshots de volume permitem capturar o estado de "
     "um banco de dados ou de um repositório de modelos em um ponto no tempo, com pouco "
     "impacto de desempenho. Combinados com replicação para outro site, formam a base de "
     "uma estratégia de recuperação de desastres com RPO e RTO definidos. Em Kubernetes, "
     "o CSI expõe snapshots como objetos VolumeSnapshot."),
    ("Kubernetes e GPUs", "Em Kubernetes, GPUs são expostas por device plugins e "
     "requisitadas pelos pods como recursos. O NVIDIA GPU Operator instala drivers, o "
     "device plugin e o exportador DCGM, que publica métricas como utilização, memória e "
     "potência para o Prometheus. Recursos como MIG e time-slicing permitem dividir uma "
     "GPU entre várias cargas."),
    ("Soberania de dados", "Rodar modelos on-premises mantém prompts, documentos e respostas "
     "dentro do perímetro da empresa, o que simplifica requisitos de privacidade e "
     "regulação. A comparação de custo com APIs de nuvem depende do volume: com uso alto e "
     "constante, GPUs próprias tendem a ter custo por token menor; com uso baixo e "
     "esporádico, pagar por token pode ser mais barato."),
    ("Rede para inferência", "Para inferência em um único nó, a rede raramente é o "
     "gargalo: o tráfego por requisição é pequeno. Ela passa a importar em modelos "
     "distribuídos entre várias GPUs ou nós, com tensor ou pipeline parallelism, onde "
     "interconexões como NVLink e RDMA reduzem a latência de comunicação entre GPUs."),
    ("SLO de inferência", "Um SLO de inferência costuma combinar o p95 do tempo até o "
     "primeiro token, que define a sensação de resposta imediata, com o p95 da latência "
     "entre tokens, que define a fluidez da leitura. A capacidade útil de um endpoint é o "
     "número de usuários simultâneos que ele atende sem violar esses limites, não a vazão "
     "máxima que ele alcança."),
]

SAMPLE_QUESTIONS = [
    "O que acontece quando o KV cache enche?",
    "Qual a diferença entre prefill e decode?",
    "Por que a vazão agregada sobe e a velocidade individual cai?",
    "Como o prefix caching afeta o TTFT?",
    "Por que o storage influencia o cold start de um modelo?",
    "Como snapshots ajudam na recuperação de desastres?",
    "Quais métricas o DCGM publica?",
    "Quando GPUs próprias são mais baratas que API de nuvem?",
    "Quando a rede vira gargalo na inferência?",
    "Como definir a capacidade útil de um endpoint de inferência?",
]


def chunk_text(text: str, size: int = 120, overlap: int = 20) -> list:
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - overlap, step)]


class RagIndex:
    def __init__(self):
        self.chunks: list = []
        self.matrix: np.ndarray | None = None
        self.stats: dict = {}

    def build(self, cfg: dict, model: str, docs: list, chunk_size: int = 120,
              batch: int = 32) -> dict:
        self.chunks = [{"doc": title, "text": c}
                       for title, text in docs for c in chunk_text(text, chunk_size)]
        if not self.chunks:
            raise ValueError("Corpus vazio")
        t0 = time.perf_counter()
        vecs = []
        for i in range(0, len(self.chunks), batch):
            vecs.extend(embed(cfg, [c["text"] for c in self.chunks[i:i + batch]], model))
        elapsed = time.perf_counter() - t0
        m = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        self.matrix = m / np.where(norms == 0, 1, norms)
        self.stats = {"chunks": len(self.chunks), "dim": int(m.shape[1]),
                      "seconds": elapsed, "chunks_per_s": len(self.chunks) / elapsed
                      if elapsed > 0 else None, "docs": len(docs)}
        return self.stats

    def search(self, qvec, k: int) -> list:
        q = np.asarray(qvec, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1)
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


def build_rag_prompt(question: str, hits: list) -> str:
    ctx = "\n\n".join(f"[{i + 1}] ({h['doc']}) {h['text']}" for i, h in enumerate(hits))
    return ("Responda à pergunta usando apenas o contexto abaixo. Cite as fontes "
            f"pelo número.\n\nContexto:\n{ctx}\n\nPergunta: {question}")


def run_query(cfg_chat: dict, cfg_emb: dict, emb_model: str, index: RagIndex,
              question: str, k: int, max_tokens: int, keep_text: bool = False) -> dict:
    rec = {"question": question, "ok": False}
    t0 = time.perf_counter()
    try:
        qvec = embed(cfg_emb, [question], emb_model)[0]
        t1 = time.perf_counter()
        hits = index.search(qvec, k)
        t2 = time.perf_counter()
        body = chat_body(cfg_chat, [{"role": "user",
                                     "content": build_rag_prompt(question, hits)}],
                         max_tokens, stream=True)
        metrics = {"start": t2}
        resp, retries = post_with_retry(cfg_chat, body, stream=True)
        text = "".join(stream_response(resp, metrics))
        meta = finalize_metrics(metrics, cfg_chat["model"], cfg_chat["prices"])
        t3 = metrics["end"]
        ttft = meta.get("ttft") or 0
        rec.update(
            ok=True,
            embed_ms=(t1 - t0) * 1000,
            retrieve_ms=(t2 - t1) * 1000,
            prefill_ms=ttft * 1000,
            decode_ms=max(t3 - t2 - ttft, 0) * 1000,
            user_ttft_ms=(t2 - t0 + ttft) * 1000,
            total_ms=(t3 - t0) * 1000,
            tokens_in=meta.get("in"), tokens_out=meta.get("out"),
            cost=meta.get("cost", 0.0), retries=retries,
            sources=[h["doc"] for h in hits],
            top_score=hits[0]["score"] if hits else None,
        )
        if keep_text:
            rec["answer"] = text
            rec["hits"] = hits
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)[:200]
        rec["total_ms"] = (time.perf_counter() - t0) * 1000
    return rec


STAGES = [("embed_ms", "Embedding"), ("retrieve_ms", "Busca vetorial"),
          ("prefill_ms", "Prefill (TTFT do modelo)"), ("decode_ms", "Geração (decode)")]


def run_rag_bench(cfg_chat, cfg_emb, emb_model, index, questions, n, concurrency,
                  k, max_tokens, on_progress, seed: int = 7) -> list:
    rng = random.Random(seed)
    qs = [rng.choice(questions) for _ in range(n)]
    out = []
    with cf.ThreadPoolExecutor(max_workers=max(concurrency, 1)) as ex:
        futs = [ex.submit(run_query, cfg_chat, cfg_emb, emb_model, index, q, k, max_tokens)
                for q in qs]
        for i, f in enumerate(cf.as_completed(futs), 1):
            out.append(f.result())
            on_progress(i, n)
    return out


def summarize_rag(records: list) -> dict:
    ok = [r for r in records if r.get("ok")]
    s = {"requests": len(records), "errors": len(records) - len(ok), "stages": []}
    for key, label in STAGES + [("user_ttft_ms", "TTFT percebido (embedding + busca + prefill)"),
                                ("total_ms", "Total")]:
        vals = [r[key] for r in ok if r.get(key) is not None]
        s["stages"].append({
            "key": key, "Etapa": label,
            "p50 (ms)": round(pct(vals, 50), 1) if vals else None,
            "p95 (ms)": round(pct(vals, 95), 1) if vals else None,
            "média (ms)": round(statistics.fmean(vals), 1) if vals else None,
        })
    tin = [r["tokens_in"] for r in ok if r.get("tokens_in")]
    s["tokens_in_mean"] = statistics.fmean(tin) if tin else None
    return s
''',
    'appai.report': r'''"""Relatório HTML autocontido de uma execução de benchmark, para enviar ao
cliente depois da demo. Abre offline (plotly.js embutido) e o navegador
exporta para PDF (Ctrl+P)."""

import html
import time

from appai import charts
from appai.bench import capacity, summarize_run
from appai.telemetry import diagnose

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:24px auto;
padding:0 16px;color:#1f2330;background:#fff}
h1{margin-bottom:4px}h2{margin-top:32px;border-bottom:2px solid #7c6ff7;padding-bottom:4px}
.sub{color:#5b6070}.hero{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}
.card{flex:1;min-width:180px;border:1px solid #e3e4ec;border-radius:10px;padding:12px 16px}
.card b{display:block;font-size:26px;color:#3b2fd1}.card span{color:#5b6070;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #e3e4ec;
padding:4px 6px;text-align:right}th{background:#f4f4fb}td:first-child,th:first-child{text-align:left}
li{margin:6px 0}.note{font-size:12px;color:#5b6070}
@media print{.plot{page-break-inside:avoid}}
"""


def _kv_table(d: dict) -> str:
    rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
                   for k, v in d.items() if v not in (None, ""))
    return f"<table>{rows}</table>"


def sweep_report(run: dict, slo: dict, title: str = "") -> str:
    info, data = run["info"], run["data"]
    mode = info.get("mode", "closed")
    levels = data["levels"]
    rows = summarize_run(levels, slo)
    cap = capacity(rows)
    unit = charts.load_unit(mode)
    infra = data.get("infra") or {}
    findings = diagnose(rows, infra.get("per_level", []), unit)

    figs = [charts.throughput_latency(rows, mode, slo),
            charts.box_by_level(levels, "ttft", "Distribuição de TTFT (ms)", charts.PURPLE,
                                1000, mode),
            charts.box_by_level(levels, "tps", "Velocidade por requisição (tok/s)",
                                charts.TEAL, 1, mode)]
    tl = charts.infra_timeline(levels, infra.get("vllm", []),
                               infra.get("prom", {}).get("series", {}))
    if tl:
        figs.insert(1, tl)
    plots = "".join(
        f'<div class="plot">{f.to_html(full_html=False, include_plotlyjs=(i == 0))}</div>'
        for i, f in enumerate(figs))

    if cap["max_ok"] is not None:
        head = f"{cap['max_ok']:g} {unit}"
        sub = ("todos os níveis testados ficaram no SLO" if cap["all_pass"]
               else f"o SLO falhou a partir de {cap['first_fail']:g}")
    else:
        head, sub = "—", "nenhum nível testado cumpriu o SLO"
    peak = max((r["throughput"] or 0 for r in rows), default=0)
    hero = (
        f'<div class="hero"><div class="card"><span>Capacidade dentro do SLO</span>'
        f'<b>{html.escape(head)}</b><span>{html.escape(sub)}</span></div>'
        f'<div class="card"><span>Vazão de pico</span><b>{peak:,.0f} tok/s</b>'
        f'<span>tokens de saída agregados</span></div>'
        f'<div class="card"><span>SLO</span><b>{slo["ttft_ms"]:.0f} ms</b>'
        f'<span>TTFT p95 · ITL p95 ≤ {slo["itl_ms"]:.0f} ms · erros ≤ '
        f'{slo["max_err_pct"]:.1f}%</span></div></div>')

    table = charts.rows_df(rows, mode).to_html(index=False, na_rep="—", border=0)
    fl = "".join(f"<li>{html.escape(f['text'])}</li>" for f in findings) \
        or "<li>Nenhum achado automático (sem telemetria ou sem anomalias).</li>"
    meta = {
        "Rótulo": run.get("label"), "Endpoint": info.get("endpoint"),
        "Modelo": info.get("model"), "Modo de carga": info.get("mode_label"),
        "Workload": info.get("workload"), "Níveis": info.get("levels"),
        "Prefix cache evitado": info.get("prefix_cache_bypass"),
        "Saída forçada (ignore_eos)": info.get("force_output"),
        "Data": info.get("timestamp"), "Versão do app": info.get("app"),
    }
    return f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title or "Relatório de benchmark")}</title><style>{CSS}</style></head><body>
<h1>{html.escape(title or "Relatório de benchmark de inferência")}</h1>
<div class="sub">{html.escape(str(info.get('model')))} · {html.escape(str(info.get('timestamp')))}</div>
{hero}
<h2>Achados</h2><ul>{fl}</ul>
<p class="note">Os achados descrevem o que foi observado na mesma janela de tempo; correlação
não implica causalidade.</p>
<h2>Gráficos</h2>{plots}
<h2>Resultados por nível</h2>{table}
<h2>Configuração</h2>{_kv_table(meta)}
<p class="note">Gerado por AI Infra Benchmark em {time.strftime('%Y-%m-%d %H:%M')}. Custos de
cloud usam preços de referência informados pelo usuário, não tabelas oficiais.</p>
</body></html>"""
''',
    'appai.store': r'''"""Histórico de execuções em SQLite.

Local: ./data/appai.db. Container/Kubernetes: APPAI_DATA_DIR=/data (PVC).
Se o diretório não for gravável, o histórico fica desabilitado e o app segue.
"""

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

KINDS = {"sweep": "Benchmark", "coldstart": "Cold start", "rag": "RAG"}


def data_dir() -> Path:
    return Path(os.environ.get("APPAI_DATA_DIR") or Path.cwd() / "data")


class Store:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else data_dir() / "appai.db"
        self.error = ""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._conn() as c:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS runs (
                        id TEXT PRIMARY KEY,
                        created_at REAL NOT NULL,
                        kind TEXT NOT NULL,
                        label TEXT NOT NULL DEFAULT '',
                        endpoint TEXT,
                        model TEXT,
                        info TEXT,
                        data TEXT
                    )""")
                c.execute("CREATE INDEX IF NOT EXISTS runs_created ON runs(created_at)")
            self.enabled = True
        except (OSError, sqlite3.Error) as e:
            self.enabled = False
            self.error = str(e)

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def save(self, kind: str, label: str, info: dict, data: dict) -> str | None:
        if not self.enabled:
            return None
        rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        with self._conn() as c:
            c.execute(
                "INSERT INTO runs (id, created_at, kind, label, endpoint, model, info, data)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (rid, time.time(), kind, label or "", info.get("endpoint"),
                 info.get("model"), json.dumps(info, default=str),
                 json.dumps(data, default=str)),
            )
        return rid

    def list(self, kind: str | None = None, limit: int = 200) -> list:
        if not self.enabled:
            return []
        q = "SELECT id, created_at, kind, label, endpoint, model, info FROM runs"
        args: tuple = ()
        if kind:
            q += " WHERE kind = ?"
            args = (kind,)
        q += " ORDER BY created_at DESC LIMIT ?"
        with self._conn() as c:
            rows = c.execute(q, args + (limit,)).fetchall()
        return [{**dict(r), "info": json.loads(r["info"] or "{}")} for r in rows]

    def get(self, rid: str) -> dict | None:
        if not self.enabled:
            return None
        with self._conn() as c:
            r = c.execute("SELECT * FROM runs WHERE id = ?", (rid,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["info"] = json.loads(d["info"] or "{}")
        d["data"] = json.loads(d["data"] or "{}")
        return d

    def rename(self, rid: str, label: str):
        if self.enabled:
            with self._conn() as c:
                c.execute("UPDATE runs SET label = ? WHERE id = ?", (label, rid))

    def delete(self, rid: str):
        if self.enabled:
            with self._conn() as c:
                c.execute("DELETE FROM runs WHERE id = ?", (rid,))


def display_name(run: dict) -> str:
    when = time.strftime("%d/%m %H:%M", time.localtime(run["created_at"]))
    label = run.get("label") or run.get("model") or run["id"]
    return f"{label} · {KINDS.get(run['kind'], run['kind'])} · {when}"
''',
    'appai.telemetry': r'''"""Telemetria de infraestrutura durante o benchmark.

Duas fontes, ambas opcionais:
- **vLLM /metrics** (formato Prometheus), raspado a cada N segundos enquanto o
  benchmark roda: fila, requisições em execução, uso de KV cache, preempções e
  prefix cache.
- **Prometheus HTTP API** (`query_range`) consultado ao final, para DCGM
  (utilização, memória e potência de GPU) ou qualquer PromQL.

Princípio: correlação temporal NÃO é causalidade. O diagnóstico descreve o que
foi observado na mesma janela e usa linguagem cautelosa.
"""

import re
import statistics
import threading
import time

import requests

# Nomes mudaram entre versões do vLLM; o primeiro encontrado vence.
VLLM_GAUGES = {
    "running": ["vllm:num_requests_running"],
    "waiting": ["vllm:num_requests_waiting"],
    "kv": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
}
VLLM_COUNTERS = {
    "preempt": ["vllm:num_preemptions_total", "vllm:num_preemptions"],
    "prefix_hits": ["vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"],
    "prefix_queries": ["vllm:prefix_cache_queries_total",
                       "vllm:gpu_prefix_cache_queries_total"],
}

DEFAULT_PROM_QUERIES = (
    "GPU util (%) | avg(DCGM_FI_DEV_GPU_UTIL)\n"
    "GPU mem (GiB) | sum(DCGM_FI_DEV_FB_USED) / 1024\n"
    "GPU power (W) | sum(DCGM_FI_DEV_POWER_USAGE)"
)

_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)')
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prom_text(text: str) -> dict:
    """Formato de exposição Prometheus → {nome: [(labels, valor), ...]}."""
    out: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, labels_raw, value = m.groups()
        try:
            v = float(value)
        except ValueError:
            continue
        labels = dict(_LABEL.findall(labels_raw or ""))
        out.setdefault(name, []).append((labels, v))
    return out


def _pick(parsed: dict, names: list, agg) -> float | None:
    for n in names:
        if n in parsed:
            return agg([v for _, v in parsed[n]])
    return None


def extract_vllm(parsed: dict) -> dict:
    """Soma entre réplicas/engines; para KV cache usa o máximo (o gargalo)."""
    s = {}
    for key, names in VLLM_GAUGES.items():
        s[key] = _pick(parsed, names, max if key == "kv" else sum)
    for key, names in VLLM_COUNTERS.items():
        s[key] = _pick(parsed, names, sum)
    return s


def scrape_vllm(url: str, verify: bool, headers: dict | None = None,
                timeout: float = 5) -> dict:
    r = requests.get(url, timeout=timeout, verify=verify, headers=headers or {})
    r.raise_for_status()
    return extract_vllm(parse_prom_text(r.text))


class VllmSampler:
    """Raspa o /metrics em uma thread enquanto o benchmark roda."""

    def __init__(self, url: str, interval: float = 1.0, verify: bool = False,
                 headers: dict | None = None):
        self.url, self.interval, self.verify = url, max(interval, 0.2), verify
        self.headers = headers or {}
        self.samples: list = []
        self.errors = 0
        self.last_error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self):
        while not self._stop.is_set():
            t = time.time()
            try:
                s = scrape_vllm(self.url, self.verify, self.headers)
                s["t"] = t
                self.samples.append(s)
            except Exception as e:  # noqa: BLE001
                self.errors += 1
                self.last_error = str(e)[:200]
            self._stop.wait(max(self.interval - (time.time() - t), 0.05))

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> list:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 6)
        return self.samples


def parse_prom_queries(text: str) -> list:
    """Linhas "Nome | PromQL" (ou só PromQL) → [(nome, query)]."""
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            name, q = line.split("|", 1)
            out.append((name.strip(), q.strip()))
        else:
            out.append((line, line))
    return out


def prom_query_range(base: str, query: str, start: float, end: float, step: float,
                     verify: bool, headers: dict | None = None) -> list:
    """Devolve [(t, valor)] — média entre séries, se a query trouxer várias."""
    r = requests.get(
        base.rstrip("/") + "/api/v1/query_range",
        params={"query": query, "start": f"{start:.3f}", "end": f"{end:.3f}",
                "step": f"{max(step, 1):.0f}s"},
        timeout=15, verify=verify, headers=headers or {},
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != "success":
        raise RuntimeError(payload.get("error", "query falhou"))
    by_t: dict = {}
    for series in payload.get("data", {}).get("result", []):
        for t, v in series.get("values", []):
            try:
                by_t.setdefault(float(t), []).append(float(v))
            except ValueError:
                continue
    return [(t, statistics.fmean(vs)) for t, vs in sorted(by_t.items())]


def collect_prometheus(base: str, queries: list, start: float, end: float,
                       step: float, verify: bool, headers: dict | None = None) -> dict:
    series, errors = {}, {}
    for name, q in queries:
        try:
            series[name] = prom_query_range(base, q, start - step, end + step, step,
                                            verify, headers)
        except Exception as e:  # noqa: BLE001
            errors[name] = str(e)[:200]
    return {"series": series, "errors": errors}


# ── Agregação por janela de nível ─────────────────────────────────────────────
def _in_window(points, t0, t1):
    return [p for p in points if t0 <= p[0] < t1]


def aggregate_levels(levels: list, vllm_samples: list, prom_series: dict) -> list:
    """Para cada nível (janela t0–t1), resume a telemetria observada."""
    out = []
    for lv in levels:
        t0, t1 = lv["t0"], lv["t1"]
        agg: dict = {"load": lv["load"]}
        win = [s for s in vllm_samples if t0 <= s["t"] < t1]
        if win:
            for key in ("waiting", "running", "kv"):
                vals = [s[key] for s in win if s.get(key) is not None]
                if vals:
                    agg[f"{key}_max"] = max(vals)
                    agg[f"{key}_mean"] = statistics.fmean(vals)
            # contadores: delta entre a amostra antes/depois da janela
            before = [s for s in vllm_samples if s["t"] <= t0] or win[:1]
            after = [s for s in vllm_samples if s["t"] >= t1] or win[-1:]
            for key in ("preempt", "prefix_hits", "prefix_queries"):
                a, b = before[-1].get(key), after[0].get(key)
                if a is not None and b is not None:
                    agg[f"{key}_delta"] = max(b - a, 0)
            q = agg.get("prefix_queries_delta")
            if q:
                agg["prefix_hit_rate"] = agg.get("prefix_hits_delta", 0) / q
        for name, pts in prom_series.items():
            vals = [v for _, v in _in_window(pts, t0, t1)]
            if vals:
                agg[f"prom:{name}:mean"] = statistics.fmean(vals)
                agg[f"prom:{name}:max"] = max(vals)
        out.append(agg)
    return out


def find_power_series(prom_series: dict) -> list | None:
    for name, pts in prom_series.items():
        low = name.lower()
        if "power" in low or "potência" in low or "(w)" in low:
            return pts
    return None


def energy_kwh(power_points: list, t0: float, t1: float) -> float | None:
    """Integra potência (W) no tempo (trapézios) → kWh na janela."""
    pts = [p for p in power_points or [] if t0 <= p[0] <= t1]
    if len(pts) < 2:
        if pts:
            return pts[0][1] * (t1 - t0) / 3.6e6
        return None
    joules = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(pts, pts[1:]))
    return joules / 3.6e6


# ── Diagnóstico com linguagem cautelosa ───────────────────────────────────────
def _prom_val(agg: dict, needle: str, stat: str = "mean"):
    for k, v in agg.items():
        if k.startswith("prom:") and needle in k.lower() and k.endswith(":" + stat):
            return v
    return None


def diagnose(rows: list, infra: list, unit: str = "usuários simultâneos") -> list:
    """Achados observacionais por nível. Cada item: {level, kind, text}.
    Nunca afirma causa: descreve o que apareceu na mesma janela."""
    by_load = {a["load"]: a for a in infra}
    findings = []
    prev_row = None
    first_fail_done = False
    for row in sorted(rows, key=lambda r: r["load"]):
        a = by_load.get(row["load"], {})
        L = row["load"]
        kv = a.get("kv_max")
        waiting = a.get("waiting_max")
        util = _prom_val(a, "util")
        if not row["slo_pass"] and not first_fail_done:
            first_fail_done = True
            parts = [f"Em {L:g} {unit} o SLO deixou de ser cumprido "
                     f"(TTFT p95 {row['ttft_p95'] or 0:.0f} ms, ITL p95 "
                     f"{row['itl_p95'] or 0:.1f} ms, erros {row['err_pct']:.1f}%)."]
            if waiting and waiting >= 1:
                parts.append(f"Na mesma janela havia até {waiting:.0f} requisição(ões) "
                             "na fila do servidor, o que pode estar relacionado ao "
                             "aumento do TTFT (o batch máximo do vLLM — `max_num_seqs` "
                             "— ou a memória de KV cache podem ter sido atingidos).")
            if util is not None and util < 50:
                parts.append(f"A GPU teve utilização média de {util:.0f}%: o gargalo "
                             "pode estar fora da GPU (rede, CPU, gateway/LB, limite "
                             "de concorrência).")
            findings.append({"level": L, "kind": "slo", "text": " ".join(parts)})
        if kv is not None and kv >= 0.9:
            findings.append({"level": L, "kind": "kv", "text":
                f"Em {L:g} {unit} o KV cache chegou a {kv * 100:.0f}%. Com o cache "
                "cheio, o vLLM passa a enfileirar ou preemptar requisições; mais "
                "memória de GPU, KV cache em FP8 ou `max_model_len` menor podem "
                "aumentar a concorrência suportada."})
        pre = a.get("preempt_delta")
        if pre:
            findings.append({"level": L, "kind": "preempt", "text":
                f"Em {L:g} {unit} houve {pre:.0f} preempção(ões): requisições foram "
                "interrompidas e recomputadas por falta de KV cache — sinal de "
                "pressão de memória que costuma elevar a latência de cauda."})
        if prev_row and prev_row.get("throughput") and row.get("throughput"):
            gain = row["throughput"] / prev_row["throughput"] - 1
            load_gain = L / prev_row["load"] - 1 if prev_row["load"] else 0
            if load_gain >= 0.5 and gain < 0.1:
                txt = (f"De {prev_row['load']:g} para {L:g} {unit} a vazão agregada "
                       f"variou só {int(round(gain * 100)):+d}%: o endpoint parece ter atingido "
                       "a saturação; carga extra vira fila e latência.")
                if util is not None and util >= 85:
                    txt += f" A GPU estava em {util:.0f}% de utilização média."
                findings.append({"level": L, "kind": "plateau", "text": txt})
        hit = a.get("prefix_hit_rate")
        if hit is not None and hit >= 0.3:
            findings.append({"level": L, "kind": "prefix", "text":
                f"Em {L:g} {unit} {hit * 100:.0f}% dos blocos de prompt vieram do "
                "prefix cache: prompts com início em comum reduzem o custo de "
                "prefill e o TTFT."})
        if row.get("lag_p95") and row["lag_p95"] > 250:
            findings.append({"level": L, "kind": "client", "text":
                f"Em {L:g} {unit} o próprio gerador atrasou os envios (p95 "
                f"{row['lag_p95']:.0f} ms). Os números desse nível podem estar "
                "limitados pelo cliente, não pelo endpoint."})
        prev_row = row
    return findings
''',
    'appai.ui': r'''''',
    'appai.ui.bench': r'''import json
import time

import pandas as pd
import streamlit as st

from appai import charts
from appai.bench import MODES, capacity, plan_total, run_sweep, summarize_run
from appai.report import sweep_report
from appai.telemetry import (VllmSampler, aggregate_levels, collect_prometheus, diagnose,
                             energy_kwh, find_power_series, parse_prom_queries, scrape_vllm)
from appai.ui.state import get_store, missing_key, slo, snapshot
from appai.version import build_info
from appai.workload import DISTRIBUTIONS, PROFILES, dataset_stats, load_dataset

CONC_LEVELS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]


def apply_profile():
    p = PROFILES[st.session_state.bench_profile]
    st.session_state.bench_in, st.session_state.bench_out = p["in"], p["out"]


def parse_rates(text: str) -> list:
    out = []
    for part in (text or "").replace(";", ",").split(","):
        try:
            v = float(part.strip())
        except ValueError:
            continue
        if v > 0:
            out.append(v)
    return sorted(set(out))


def prom_headers() -> dict:
    tok = st.session_state.tele_prom_token.strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


# ── Configuração ──────────────────────────────────────────────────────────────
def render_workload():
    s = st.session_state
    st.radio("Modo de carga", list(MODES), format_func=MODES.get, key="bench_mode",
             horizontal=True,
             help="Closed-loop: N usuários que só mandam a próxima pergunta quando a "
                  "anterior termina. Open-loop: requisições chegam numa taxa (Poisson), "
                  "como tráfego real — se o servidor não acompanha, a fila cresce.")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.selectbox("Perfil de carga", list(PROFILES), key="bench_profile",
                     on_change=apply_profile)
        st.slider("Tokens de entrada (média aprox.)", 16, 8000, key="bench_in", step=16)
        st.slider("Tokens de saída (max_tokens, média)", 16, 4096, key="bench_out", step=16)
        st.selectbox("Distribuição de tamanhos", DISTRIBUTIONS, key="bench_dist",
                     help="Lognormal: tamanhos variados em torno da média, com cauda "
                          "longa — mais próximo de tráfego real que prompts idênticos.")
        if s.bench_dist.startswith("Lognormal"):
            st.slider("Variação (coef. de variação)", 0.1, 2.0, key="bench_cv", step=0.1)
    with b2:
        if s.bench_mode == "open":
            st.text_input("Taxas de chegada (req/s, separadas por vírgula)", key="bench_rates")
            st.slider("Duração por nível (s)", 5, 300, key="bench_duration", step=5)
        else:
            st.multiselect("Níveis de concorrência", CONC_LEVELS, key="bench_levels")
            st.slider("Rodadas por nível", 1, 10, key="bench_rounds",
                      help="Requisições por worker: total do nível = concorrência × rodadas.")
        st.toggle("Evitar prefix cache (prefixo único por requisição)", key="bench_nocache")
        st.toggle("Forçar tamanho de saída (`ignore_eos`, vLLM)", key="bench_force_output",
                  help="Gera exatamente max_tokens. Sem isso o modelo pode parar antes e "
                       "a carga fica menor do que parece. Servidores que não são vLLM "
                       "podem rejeitar o parâmetro.")
        st.toggle("Aquecimento antes de medir (2 requisições descartadas)", key="bench_warmup")
    with b3:
        up = st.file_uploader("Dataset de prompts reais (opcional)",
                              type=["jsonl", "json", "txt"],
                              help="JSONL com \"prompt\" (e opcional \"max_tokens\"), "
                                   "ShareGPT (.json) ou TXT com um prompt por linha.")
        if up is not None and up.name != s.bench_dataset_name:
            try:
                s.bench_dataset = load_dataset(up.getvalue(), up.name)
                s.bench_dataset_name = up.name
                s.bench_use_dataset = bool(s.bench_dataset)
            except Exception as e:
                st.error(f"Não consegui ler o dataset: {e}")
        if s.bench_dataset:
            st.toggle(f"Usar dataset `{s.bench_dataset_name}`", key="bench_use_dataset")
            ds = dataset_stats(s.bench_dataset)
            st.caption(f"{ds['n']} prompts · entrada mediana ~{ds['in_median']} tokens "
                       f"(máx. {ds['in_max']}) · saída conhecida em {ds['out_known']}")


def render_slo():
    st.markdown("##### 🎯 SLO (alvo de qualidade)")
    c1, c2, c3 = st.columns(3)
    c1.number_input("TTFT p95 máximo (ms)", key="slo_ttft", min_value=50.0, step=50.0,
                    help="Tempo até o primeiro token: a sensação de resposta imediata.")
    c2.number_input("ITL p95 máximo (ms)", key="slo_itl", min_value=5.0, step=5.0,
                    help="Latência entre tokens: a fluidez da leitura. 50 ms ≈ 20 tok/s.")
    c3.number_input("Erros máximos (%)", key="slo_err", min_value=0.0, step=0.5)
    st.caption("A capacidade é recalculada na hora quando o SLO muda — não precisa rodar "
               "de novo.")


def render_telemetry_cfg():
    s = st.session_state
    with st.expander("📡 Telemetria de infra (opcional): vLLM /metrics e Prometheus/DCGM",
                     expanded=bool(s.tele_vllm_url or s.tele_prom_url)):
        t1, t2 = st.columns(2)
        with t1:
            st.text_input("URL do /metrics do vLLM", key="tele_vllm_url",
                          placeholder="http://<pod-ou-service-do-modelo>:8000/metrics",
                          help="Raspado durante o benchmark: fila, requisições em execução, "
                               "KV cache, preempções e prefix cache.")
            st.slider("Intervalo de coleta (s)", 0.5, 10.0, key="tele_interval", step=0.5)
            if st.button("Testar /metrics", disabled=not s.tele_vllm_url):
                try:
                    st.json(scrape_vllm(s.tele_vllm_url, s.verify_tls))
                except Exception as e:
                    st.error(f"Falhou: {e}")
        with t2:
            st.text_input("URL do Prometheus", key="tele_prom_url",
                          placeholder="http://prometheus.monitoring:9090",
                          help="Consultado ao final (query_range) na janela do benchmark.")
            st.text_input("Bearer token do Prometheus (opcional)", key="tele_prom_token",
                          type="password")
        st.text_area("Consultas PromQL (uma por linha: `Nome | PromQL`)",
                     key="tele_prom_queries", height=100,
                     help="Padrão: métricas DCGM do NVIDIA GPU Operator. Use '(W)' no nome "
                          "da série de potência para o cálculo de energia.")
        st.caption("Em NKP/NAI o /metrics fica no pod de serving do modelo (porta do vLLM). "
                   "Se o app roda fora do cluster, use `kubectl port-forward` ou o "
                   "Prometheus que já coleta o vLLM.")


# ── Execução ──────────────────────────────────────────────────────────────────
def build_plan() -> dict:
    s = st.session_state
    profile = PROFILES[s.bench_profile]
    spec = {"question": profile["q"], "in": int(s.bench_in), "out": int(s.bench_out),
            "dist": s.bench_dist, "cv": float(s.bench_cv), "nocache": s.bench_nocache,
            "dataset": s.bench_dataset if (s.bench_use_dataset and s.bench_dataset) else None}
    if s.bench_mode == "open":
        levels = parse_rates(s.bench_rates)
    else:
        levels = sorted(s.bench_levels)
    return {"mode": s.bench_mode, "levels": levels, "rounds": int(s.bench_rounds),
            "duration_s": int(s.bench_duration), "spec": spec, "warmup": s.bench_warmup,
            "seed": int(time.time())}


def workload_desc(plan: dict) -> str:
    sp = plan["spec"]
    if sp.get("dataset"):
        return f"dataset {st.session_state.bench_dataset_name} ({len(sp['dataset'])} prompts)"
    d = f"~{sp['in']} in / {sp['out']} out tokens"
    if sp["dist"].startswith("Lognormal"):
        d += f", lognormal cv={sp['cv']:.1f}"
    return f"{st.session_state.bench_profile}: {d}"


def execute(plan: dict):
    s = st.session_state
    extra = {"ignore_eos": True} if s.bench_force_output else {}
    cfg = snapshot(extra)
    bar = st.progress(0.0, text="Aquecendo…" if plan["warmup"] else "Iniciando…")
    unit = "req/s" if plan["mode"] == "open" else "conc."

    def on_progress(done, tot, load):
        bar.progress(min(done / tot, 1.0), text=f"{unit} {load:g}: {done}/{tot} requisições")

    sampler = None
    if s.tele_vllm_url.strip():
        sampler = VllmSampler(s.tele_vllm_url.strip(), s.tele_interval, s.verify_tls).start()
    try:
        res = run_sweep(cfg, plan, on_progress)
    finally:
        samples = sampler.stop() if sampler else []
    bar.progress(1.0, text="Coletando telemetria…")

    prom = {"series": {}, "errors": {}}
    if s.tele_prom_url.strip() and res["levels"]:
        step = max(float(s.tele_interval), 1.0)
        prom = collect_prometheus(s.tele_prom_url.strip(),
                                  parse_prom_queries(s.tele_prom_queries),
                                  res["levels"][0]["t0"], res["levels"][-1]["t1"], step,
                                  s.verify_tls, prom_headers())
    per_level = aggregate_levels(res["levels"], samples, prom["series"])
    power = find_power_series(prom["series"])
    if power:
        for agg, lv in zip(per_level, res["levels"]):
            agg["energy_kwh"] = energy_kwh(power, lv["t0"], lv["t1"])
    bar.empty()

    b = build_info()
    info = {
        "endpoint": cfg["base"], "model": cfg["model"], "mode": plan["mode"],
        "mode_label": MODES[plan["mode"]], "workload": workload_desc(plan),
        "levels": ", ".join(f"{x:g}" for x in plan["levels"]),
        "rounds": plan["rounds"], "duration_s": plan["duration_s"],
        "prefix_cache_bypass": plan["spec"]["nocache"], "force_output": bool(extra),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "app": f"v{b['version']} ({b['commit']})",
        "warmup_errors": sum(1 for r in res["warmup"] if not r.get("ok")),
    }
    data = {"levels": res["levels"], "infra": {
        "vllm": samples, "prom": prom, "per_level": per_level,
        "sampler_errors": sampler.errors if sampler else 0,
        "sampler_last_error": sampler.last_error if sampler else ""}}
    label = s.bench_label.strip() or f"{cfg['model']} · {workload_desc(plan)}"
    run = {"label": label, "info": info, "data": data, "kind": "sweep"}
    run["id"] = get_store().save("sweep", label, info, data)
    s.bench_result = run


# ── Resultados ────────────────────────────────────────────────────────────────
def capacity_headline(cap: dict, unit: str) -> tuple:
    if cap["max_ok"] is None:
        return "—", "nenhum nível testado cumpriu o SLO"
    if cap["all_pass"]:
        return f"≥ {cap['max_ok']:g}", f"{unit}: todos os níveis no SLO — teste mais carga"
    return f"{cap['max_ok']:g}", f"{unit} (falhou em {cap['first_fail']:g})"


def infra_table(per_level: list, rows: list, mode: str) -> pd.DataFrame | None:
    out_by = {r["load"]: r["tokens_out"] for r in rows}
    recs = []
    for a in per_level:
        r = {charts.load_label(mode): a["load"]}
        if "waiting_max" in a:
            r["Fila máx."] = a["waiting_max"]
        if "running_max" in a:
            r["Em execução máx."] = a["running_max"]
        if "kv_max" in a:
            r["KV cache máx. (%)"] = round(a["kv_max"] * 100, 1)
        if "preempt_delta" in a:
            r["Preempções"] = a["preempt_delta"]
        if "prefix_hit_rate" in a:
            r["Prefix cache hit (%)"] = round(a["prefix_hit_rate"] * 100, 1)
        for k, v in a.items():
            if k.startswith("prom:") and k.endswith(":mean"):
                r[k[5:-5] + " (média)"] = round(v, 1)
        if a.get("energy_kwh") and out_by.get(a["load"]):
            r["kWh / 1M tokens"] = round(a["energy_kwh"] / out_by[a["load"]] * 1e6, 3)
        if len(r) > 1:
            recs.append(r)
    return pd.DataFrame(recs) if recs else None


def render_sweep(run: dict, key: str = "cur"):
    """Mostra uma execução (atual ou do histórico) com o SLO atual."""
    info, data = run["info"], run["data"]
    mode = info.get("mode", "closed")
    unit = charts.load_unit(mode)
    levels = data["levels"]
    cur_slo = slo()
    rows = summarize_run(levels, cur_slo)
    if not rows:
        st.info("Execução sem níveis.")
        return
    cap = capacity(rows)
    head, sub = capacity_headline(cap, unit)
    peak = max(rows, key=lambda r: r["throughput"] or 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Capacidade dentro do SLO", head, help=sub)
    m1.caption(sub)
    m2.metric("Pico de vazão", f"{peak['throughput'] or 0:,.0f} tok/s",
              help=f"Atingido em {peak['load']:g} {unit}")
    m2.caption(f"em {peak['load']:g} {unit}")
    m3.metric("Goodput no limite do SLO",
              f"{cap['goodput_rps']:.2f} req/s" if cap["goodput_rps"] is not None else "—",
              help="Requisições por segundo que cumprem o SLO individualmente.")
    errs = sum(r["errors"] for r in rows)
    m4.metric("Erros / Retries", f"{errs} / {sum(r['retries'] for r in rows)}")

    st.plotly_chart(charts.throughput_latency(rows, mode, cur_slo), width="stretch",
                    key=f"{key}-tl")

    infra = data.get("infra") or {}
    per_level = infra.get("per_level") or []
    samples = infra.get("vllm") or []
    prom_series = (infra.get("prom") or {}).get("series") or {}
    findings = diagnose(rows, per_level, unit)
    if findings:
        st.markdown("##### 🔎 O que a infraestrutura mostrou")
        for f in findings:
            st.markdown(f"- {f['text']}")
        st.caption("Observações na mesma janela de tempo — correlação, não causalidade.")
    fig = charts.infra_timeline(levels, samples, prom_series)
    if fig:
        st.plotly_chart(fig, width="stretch", key=f"{key}-infra")
        df_inf = infra_table(per_level, rows, mode)
        if df_inf is not None:
            st.dataframe(df_inf, width="stretch", hide_index=True)
    for name, err in ((infra.get("prom") or {}).get("errors") or {}).items():
        st.warning(f"Prometheus `{name}`: {err}")
    if infra.get("sampler_errors"):
        st.warning(f"/metrics do vLLM: {infra['sampler_errors']} coleta(s) falharam — "
                   f"{infra.get('sampler_last_error')}")

    g1, g2 = st.columns(2)
    g1.plotly_chart(charts.box_by_level(levels, "ttft", "Distribuição de TTFT (ms)",
                                        charts.PURPLE, 1000, mode),
                    width="stretch", key=f"{key}-ttft")
    g2.plotly_chart(charts.box_by_level(levels, "tps", "Velocidade por requisição (tok/s)",
                                        charts.TEAL, 1, mode),
                    width="stretch", key=f"{key}-tps")

    df = charts.rows_df(rows, mode)
    st.dataframe(df, width="stretch", hide_index=True)
    if mode == "open" and any((r.get("lag_p95") or 0) > 250 for r in rows):
        st.warning("O gerador atrasou envios em algum nível: os números desse nível podem "
                   "estar limitados pelo cliente.")
    if any(not r.get("usage_reported", True) for lv in levels for r in lv["records"]
           if r.get("ok")):
        st.caption("⚠️ O servidor não devolveu `usage` em algumas respostas; tokens de saída "
                   "foram contados por chunk SSE.")
    st.caption(f"**{run.get('label') or ''}** · {info.get('mode_label')} · "
               f"{info.get('workload')} · modelo `{info.get('model')}` · {info.get('timestamp')}"
               + (f" · id `{run['id']}`" if run.get("id") else ""))

    d1, d2, d3, _ = st.columns([1, 1, 1.4, 2])
    d1.download_button("⬇ CSV", df.to_csv(index=False), "benchmark.csv", "text/csv",
                       key=f"{key}-csv")
    d2.download_button("⬇ JSON", json.dumps({"label": run.get("label"), "info": info,
                                             "slo": cur_slo, "summary": rows, "data": data},
                                            ensure_ascii=False, indent=1, default=str),
                       "benchmark.json", "application/json", key=f"{key}-json")
    if d3.button("📄 Gerar relatório HTML", key=f"{key}-mkrep"):
        st.session_state[f"{key}-report"] = sweep_report(run, cur_slo, run.get("label") or "")
    if st.session_state.get(f"{key}-report"):
        d3.download_button("⬇ relatório.html", st.session_state[f"{key}-report"],
                           "relatorio-benchmark.html", "text/html", key=f"{key}-rep")


def render():
    s = st.session_state
    st.caption("Mede o que importa para dimensionar inferência: TTFT e ITL (p50/p95/p99), "
               "tokens/s, vazão agregada e — principalmente — **quantos usuários o endpoint "
               "atende dentro do SLO**. Com telemetria, mostra o porquê.")
    render_workload()
    render_slo()
    render_telemetry_cfg()

    plan = build_plan()
    total = plan_total(plan) if plan["levels"] else 0
    r1, r2, r3 = st.columns([2, 1, 1])
    r1.text_input("Rótulo da execução (para o histórico e o A/B)", key="bench_label",
                  placeholder="ex.: L40S · Llama-3.1-8B · Files")
    r2.metric("Requisições nesta execução", total)
    if plan["mode"] == "open" and plan["levels"]:
        r2.caption(f"~{len(plan['levels']) * plan['duration_s']} s de carga")
    run = r3.button("▶ Executar benchmark", type="primary", width="stretch",
                    disabled=not plan["levels"], key="run_bench")
    r3.caption("⚠️ Consome tokens/GPU reais do endpoint.")

    if run:
        if missing_key():
            st.error("Insira sua API Key na sidebar.")
        else:
            execute(plan)
            if not get_store().enabled:
                st.warning("Histórico desativado: esta execução não foi salva.")

    if s.bench_result:
        st.divider()
        render_sweep(s.bench_result, "cur")
    else:
        st.info("Nenhum resultado ainda. Ajuste a carga e clique em **Executar benchmark**. "
                "Sem GPU por perto? Use o **mock embutido** na sidebar.")
''',
    'appai.ui.chat': r'''import json
import statistics
import time

import requests
import streamlit as st

from appai.client import (call_blocking, chat_body, finalize_metrics, post_with_retry,
                          redact, sanitize_messages, stream_response)
from appai.cost import fmt_cost
from appai.ui.state import missing_key, snapshot


def format_meta(meta: dict) -> str:
    parts = []
    if meta.get("tps"):
        parts.append(f"⚡ {meta['tps']:.1f} tok/s")
    if meta.get("ttft") is not None:
        parts.append(f"TTFT {meta['ttft'] * 1000:.0f} ms")
    if meta.get("itl"):
        parts.append(f"ITL {meta['itl']:.1f} ms")
    if meta.get("out"):
        parts.append(f"{meta['in']} in / {meta['out']} out tokens")
    if meta.get("cost") is not None:
        parts.append(f"💰 {fmt_cost(meta['cost'])} (cloud pública)")
    if meta.get("model"):
        parts.append(meta["model"])
    return "  ·  ".join(parts)


def session_kpis() -> dict:
    metas = [m["meta"] for m in st.session_state.messages if m.get("meta")]
    ttft = [m["ttft"] * 1000 for m in metas if m.get("ttft") is not None]
    tps = [m["tps"] for m in metas if m.get("tps")]
    return {
        "n": len(metas),
        "ttft": statistics.fmean(ttft) if ttft else None,
        "tps": statistics.fmean(tps) if tps else None,
        "tin": sum(m.get("in", 0) for m in metas),
        "tout": sum(m.get("out", 0) for m in metas),
        "cost": sum(m.get("cost", 0.0) for m in metas),
    }


def render_kpis(slot):
    """Faixa de KPIs da sessão. Chamada de novo após cada resposta do chat para
    não ficar uma rodada defasada."""
    k = session_kpis()
    with slot.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Respostas", k["n"])
        c2.metric("TTFT médio", f"{k['ttft']:.0f} ms" if k["ttft"] is not None else "—")
        c3.metric("Tokens/s médio", f"{k['tps']:.1f}" if k["tps"] is not None else "—")
        c4.metric("Tokens in / out", f"{k['tin']} / {k['tout']}")
        c5.metric("Custo cloud acumulado", fmt_cost(k["cost"]) if k["n"] else "—")


def build_messages() -> list:
    s = st.session_state
    msgs = []
    if s.system_prompt.strip():
        msgs.append({"role": "system", "content": s.system_prompt.strip()})
    msgs.extend(sanitize_messages(s.messages))
    return msgs


def render(kpi_slot):
    s = st.session_state
    for msg in s.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(format_meta(msg["meta"]))

    if s.show_debug and s.debug_log:
        with st.expander("🐛 Debug, última requisição", expanded=False):
            st.code(s.debug_log, language="http")

    prompt = st.chat_input("Digite sua mensagem…")
    if not prompt:
        return
    if missing_key():
        st.error("Insira sua API Key na sidebar.")
        return

    s.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    cfg = snapshot()
    body = chat_body(cfg, build_messages(), s.max_tokens, stream=s.use_stream)
    s.debug_log = (f"POST {cfg['base']}/chat/completions\n"
                   f"Headers: {json.dumps(redact(cfg['headers']), indent=2)}\n"
                   f"Body: {json.dumps(body, indent=2, ensure_ascii=False)}\n")
    req_log: list = []

    with st.chat_message("assistant", avatar="🤖"):
        notice = st.empty()
        try:
            if s.use_stream:
                metrics: dict = {"start": time.perf_counter()}
                resp, _ = post_with_retry(cfg, body, stream=True, notify=notice.info,
                                          log=req_log)
                notice.empty()
                content = st.write_stream(stream_response(resp, metrics))
                meta = finalize_metrics(metrics, cfg["model"], cfg["prices"])
                req_log.append(f"Status: 200 (stream)\nUsage: {json.dumps(metrics.get('usage'))}")
            else:
                with st.spinner("Aguardando resposta…"):
                    content, meta = call_blocking(cfg, body, notify=notice.info, log=req_log)
                notice.empty()
                st.markdown(content)
            st.caption(format_meta(meta))
            s.messages.append({"role": "assistant", "content": content, "meta": meta})
        except requests.exceptions.HTTPError as e:
            notice.empty()
            st.error(f"Erro HTTP {e.response.status_code} após {s.max_retries} "
                     f"tentativa(s): {e.response.text[:500]}")
        except requests.exceptions.ConnectionError as e:
            notice.empty()
            st.error(f"Endpoint inacessível: {e}")
        except Exception as e:
            notice.empty()
            st.error(f"Erro: {e}")
        finally:
            if req_log:
                s.debug_log += "\n" + "\n".join(req_log) + "\n"
    render_kpis(kpi_slot)
''',
    'appai.ui.coldstart': r'''import time

import pandas as pd
import requests
import streamlit as st

from appai import charts
from appai.coldstart import STORAGE_BACKENDS, load_throughput_gbs, measure_cold_start
from appai.ui.state import get_store, missing_key, snapshot
from appai.version import build_info


def trigger_hibernation(url: str) -> str:
    cfg = snapshot()
    r = requests.post(url, headers=cfg["headers"], timeout=15, verify=cfg["verify"])
    return f"HTTP {r.status_code}"


def summary_row(run: dict) -> dict:
    info, data = run["info"], run["data"]
    when = time.strftime("%d/%m %H:%M", time.localtime(run["created_at"]))
    return {"name": f"{run.get('label') or info.get('backend')} · {when}",
            "ready_s": data.get("ready_s"),
            "total_s": data.get("total_s")}


def render():
    s = st.session_state
    st.caption(
        "Quanto tempo um endpoint **hibernado ou escalado a zero** leva até responder. "
        "Boa parte é ler dezenas de GB de pesos do storage para a GPU — por isso cada "
        "medição é rotulada pelo backend de storage, para comparar Files × Volumes × "
        "NVMe local."
    )
    with st.expander("Como deixar o endpoint frio antes de medir", expanded=False):
        st.markdown(
            "- **NAI**: hibernar o endpoint na interface (ou deixar o pool do Unified "
            "Endpoint hibernar por inatividade).\n"
            "- **KServe / vLLM em NKP**: `minReplicas: 0` (scale-to-zero) ou "
            "`kubectl scale deploy/<modelo> --replicas=0`.\n"
            "- Para isolar o storage, repita com o mesmo modelo em PVCs de backends "
            "diferentes. Cache de página do nó também conta: o primeiro cold start "
            "depois de um reboot tende a ser o mais lento.\n"
            "- Opcional: informe abaixo uma URL que hiberna o endpoint via POST "
            "(o mock embutido tem `/admin/hibernate`)."
        )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.selectbox("Backend de storage dos pesos", STORAGE_BACKENDS, key="cs_backend")
        st.text_input("Rótulo (opcional)", key="cs_label",
                      placeholder="ex.: Llama-3.1-8B · Files · 1ª após reboot")
    with c2:
        st.number_input("Tamanho dos pesos (GB)", key="cs_model_gb", min_value=0.0, step=1.0,
                        help="Para estimar a vazão efetiva de carga (GB/s).")
        st.number_input("Timeout (s)", key="cs_timeout", min_value=10, step=30)
        st.number_input("Intervalo entre tentativas (s)", key="cs_poll", min_value=0.5,
                        step=0.5)
    with c3:
        st.text_input("URL para hibernar antes (POST, opcional)", key="cs_hibernate_url")
        run = st.button("🧊 Medir cold start", type="primary", width="stretch")
        st.caption("Envia uma requisição mínima e repete até o primeiro token.")

    if run:
        if missing_key():
            st.error("Insira sua API Key na sidebar.")
        else:
            if s.cs_hibernate_url.strip():
                try:
                    st.info(f"Hibernação solicitada: {trigger_hibernation(s.cs_hibernate_url)}")
                except Exception as e:
                    st.error(f"Falha ao hibernar: {e}")
            log = st.empty()
            lines: list = []

            def on_event(e):
                lines.append(f"{e['t']:>7.1f}s  {e['status']}  {e['note']}")
                log.code("\n".join(lines[-12:]))

            cfg = snapshot()
            with st.spinner("Aguardando o endpoint acordar…"):
                res = measure_cold_start(cfg, timeout_s=float(s.cs_timeout),
                                         poll_s=float(s.cs_poll), on_event=on_event)
            b = build_info()
            info = {"endpoint": cfg["base"], "model": cfg["model"], "backend": s.cs_backend,
                    "model_gb": s.cs_model_gb, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "app": f"v{b['version']} ({b['commit']})"}
            label = s.cs_label.strip() or f"{cfg['model']} · {s.cs_backend}"
            rid = get_store().save("coldstart", label, info, res) if res["ok"] else None
            s.cs_result = {"id": rid, "label": label, "info": info, "data": res}

    r = s.cs_result
    if r:
        d = r["data"]
        st.divider()
        if not d["ok"]:
            st.error(f"Não respondeu: {d['error']} ({d['attempts']} tentativa(s))")
        else:
            gbs = load_throughput_gbs(r["info"].get("model_gb"), d["ready_s"])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Até responder", f"{d['ready_s']:.1f} s")
            m2.metric("Até o 1º token", f"{d['total_s']:.1f} s" if d["total_s"] else "—")
            m3.metric("Tentativas", d["attempts"])
            m4.metric("Vazão efetiva de carga", f"{gbs:.2f} GB/s" if gbs else "—",
                      help="Pesos ÷ tempo até responder. Limite inferior: inclui agendamento, "
                           "pull de imagem e inicialização do runtime.")
            if d["attempts"] == 1:
                st.info("Respondeu na primeira tentativa: o endpoint provavelmente já estava "
                        "quente. Hiberne-o antes para medir o cold start.")
        if d["events"]:
            st.dataframe(pd.DataFrame(d["events"]).rename(
                columns={"t": "t (s)", "status": "Status", "note": "Detalhe"}),
                width="stretch", hide_index=True, height=200)

    past = get_store().list("coldstart", limit=20)
    if past:
        st.markdown("##### Cold starts salvos")
        runs = [summary_row(get_store().get(p["id"])) for p in reversed(past)]
        st.plotly_chart(charts.coldstart_bars(runs), width="stretch", key="cs-bars")
''',
    'appai.ui.cost': r'''import streamlit as st

from appai import charts
from appai.bench import capacity, summarize_run
from appai.cost import CLOUD_PRESETS, cost_curve, fmt_cost, onprem_analysis
from appai.store import display_name
from appai.ui.chat import session_kpis
from appai.ui.state import get_store, slo

BASES = ["Vazão dentro do SLO", "Vazão de pico"]


def apply_cloud_preset():
    preset = CLOUD_PRESETS.get(st.session_state.price_preset)
    if preset:
        st.session_state.price_in_per_1m, st.session_state.price_out_per_1m = preset


def _sources() -> dict:
    out = {}
    cur = st.session_state.bench_result
    if cur:
        out["__current__"] = ("Benchmark atual", cur)
    for r in get_store().list("sweep", limit=50):
        if cur and r["id"] == cur.get("id"):
            continue
        out[r["id"]] = (display_name(r), None)
    return out


def _measured(run: dict, basis: str) -> dict:
    """Vazão e potência medidas de uma execução (pico ou limite do SLO)."""
    rows = summarize_run(run["data"]["levels"], slo())
    cap = capacity(rows)
    if basis == BASES[0]:
        load, tps = cap["max_ok"], cap["throughput"]
    else:
        best = max(rows, key=lambda r: r["throughput"] or 0)
        load, tps = best["load"], best["throughput"]
    per_level = (run["data"].get("infra") or {}).get("per_level") or []
    agg = next((a for a in per_level if a["load"] == load), {})
    power = next((v for k, v in agg.items()
                  if k.startswith("prom:") and ("power" in k.lower() or "(w)" in k.lower())
                  and k.endswith(":mean")), None)
    row = next((r for r in rows if r["load"] == load), None)
    kwh_m = (agg["energy_kwh"] / row["tokens_out"] * 1e6
             if agg.get("energy_kwh") and row and row["tokens_out"] else None)
    return {"tps": tps or 0, "load": load, "power_w": power, "kwh_per_m": kwh_m,
            "mode": run["info"].get("mode", "closed")}


def render():
    s = st.session_state
    st.caption(
        "Compara o custo por token em uma cloud pública com o custo de rodar a mesma carga "
        "on-prem. **Preços, custo de GPU, utilização e energia são premissas editáveis, não "
        "medições** — a vazão (e a potência, com DCGM) vem do benchmark."
    )
    st.markdown("##### ☁️ Cloud pública (US$ por 1M de tokens)")
    p1, p2, p3 = st.columns(3)
    p1.selectbox("Referência de preço", list(CLOUD_PRESETS), key="price_preset",
                 on_change=apply_cloud_preset,
                 help="Valores ilustrativos — não são a tabela de preços de nenhum provedor.")
    p2.number_input("Entrada (US$ / 1M tokens)", key="price_in_per_1m", min_value=0.0,
                    step=0.1, format="%.2f")
    p3.number_input("Saída (US$ / 1M tokens)", key="price_out_per_1m", min_value=0.0,
                    step=0.1, format="%.2f")

    k = session_kpis()
    if k["n"]:
        avg = k["cost"] / k["n"]
        q1, q2, q3 = st.columns(3)
        q1.metric("Custo médio por troca de mensagem", fmt_cost(avg))
        q2.metric("Custo desta sessão", fmt_cost(k["cost"]))
        q3.metric("Projeção: 1.000 trocas", f"US$ {avg * 1000:.2f}")

    st.markdown("##### 🖥️ On-prem (GPU própria) × Cloud")
    src = _sources()
    if not src:
        st.info("Rode o **Benchmark** para usar a vazão medida no cálculo on-prem.")
        return
    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Vazão medida de", list(src), format_func=lambda x: src[x][0])
    c2.radio("Base de capacidade", BASES, key="cost_basis",
             help="Dentro do SLO: a vazão no maior nível que ainda cumpre o SLO — o que "
                  "o endpoint entrega com qualidade. Pico: o máximo absoluto.")
    run = src[choice][1] or get_store().get(choice)
    meas = _measured(run, s.cost_basis)

    o1, o2, o3 = st.columns(3)
    o1.number_input("Custo por GPU-hora (US$)", key="gpu_hour", min_value=0.0, step=0.5,
                    format="%.2f", help="Amortização do hardware + operação (+ energia, se "
                                        "não for separada abaixo).")
    o2.number_input("Nº de GPUs do endpoint", key="gpu_count", min_value=1, step=1)
    o3.slider("Utilização média (%)", 5, 100, key="utilization_pct",
              help="Fração do mês em que a GPU trabalha na vazão medida. 100% é o teto "
                   "teórico; tráfego real tem picos e vales.")
    v1, v2 = st.columns(2)
    v1.number_input("Volume mensal — tokens de entrada (milhões)", key="monthly_in_m",
                    min_value=0.0, step=100.0)
    v2.number_input("Volume mensal — tokens de saída (milhões)", key="monthly_out_m",
                    min_value=0.0, step=50.0)
    with st.expander("⚡ Energia", expanded=s.energy_separate):
        st.toggle("Calcular energia à parte (o custo de GPU-hora não inclui energia)",
                  key="energy_separate")
        e1, e2, e3 = st.columns(3)
        e1.number_input("Potência média por GPU (W)", key="power_w", min_value=0.0, step=25.0)
        if meas["power_w"]:
            per_gpu = meas["power_w"] / max(int(s.gpu_count), 1)
            e1.caption(f"Medido (DCGM): {meas['power_w']:.0f} W no total ≈ {per_gpu:.0f} W/GPU")
        e2.number_input("PUE do datacenter", key="pue", min_value=1.0, step=0.05)
        e3.number_input("Preço da energia (US$/kWh)", key="kwh_price", min_value=0.0,
                        step=0.01, format="%.3f")
        if meas["kwh_per_m"]:
            st.caption(f"Energia medida no nível usado: **{meas['kwh_per_m']:.3f} kWh por 1M "
                       "tokens de saída** (só GPU, sem PUE).")

    if not meas["tps"]:
        st.warning("Nenhum nível cumpriu o SLO nesta execução: não há vazão com qualidade "
                   "para basear o custo. Use 'Vazão de pico' ou ajuste o SLO.")
        return
    params = {"peak_tps": meas["tps"], "utilization": s.utilization_pct / 100,
              "gpu_hour": s.gpu_hour, "gpu_count": s.gpu_count,
              "monthly_in_m": s.monthly_in_m, "monthly_out_m": s.monthly_out_m,
              "price_in": s.price_in_per_1m, "price_out": s.price_out_per_1m,
              "energy_separate": s.energy_separate, "power_w": s.power_w, "pue": s.pue,
              "kwh_price": s.kwh_price}
    a = onprem_analysis(params)
    unit = charts.load_unit(meas["mode"])

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Vazão usada", f"{meas['tps']:,.0f} tok/s",
              help=f"{s.cost_basis}: {meas['load']:g} {unit}")
    r1.caption(f"{s.cost_basis.lower()} · {meas['load']:g} {unit}")
    if a["onprem_per_m_out"] is not None:
        r2.metric("On-prem: US$ / 1M tokens de saída", f"{a['onprem_per_m_out']:.2f}",
                  delta=f"{a['onprem_per_m_out'] - a['cloud_per_m_out']:+.2f} vs cloud",
                  delta_color="inverse",
                  help="Com a entrada processada junto, na proporção do benchmark.")
    r3.metric("Cloud: US$ / 1M tokens de saída", f"{a['cloud_per_m_out']:.2f}",
              help="Preço de saída + entrada na proporção do volume mensal informado.")
    r4.metric("Capacidade de 1 endpoint", f"{a['capacity_out_m']:,.0f}M tok/mês",
              help="Vazão × 730 h × utilização.")

    m1, m2, m3 = st.columns(3)
    m1.metric("On-prem / mês", f"US$ {a['onprem_month']:,.0f}",
              help=f"{a['replicas']} endpoint(s) × US$ {a['set_month']:,.0f}"
                   + (f" (energia US$ {a['energy_month']:,.0f})" if a["energy_month"] else ""))
    m1.caption(f"{a['replicas']} endpoint(s) de {s.gpu_count} GPU(s)")
    m2.metric("Cloud / mês", f"US$ {a['cloud_month']:,.0f}")
    m3.metric("Diferença / mês", f"US$ {a['savings_month']:,.0f}",
              help="Positivo: on-prem mais barato para este volume.")

    msg = ""
    if a["breakeven_m"]:
        msg = f"Break-even: **{a['breakeven_m']:,.0f}M tokens de saída/mês**"
        if not a["breakeven_within_capacity"]:
            msg += (" — acima da capacidade de um endpoint nesta utilização: com estas "
                    "premissas a cloud fica mais barata em qualquer volume.")
        else:
            msg += "."
    st.markdown(msg)
    st.plotly_chart(charts.cost_curve_fig(cost_curve(params), s.monthly_out_m),
                    width="stretch", key="cost-curve")
    st.caption(
        "Simplificações: a vazão medida embute a proporção entrada/saída do benchmark; "
        "on-prem escala em endpoints inteiros (degraus); não inclui rede, storage, "
        "licenças nem pessoas — some ao custo de GPU-hora se quiser."
    )
''',
    'appai.ui.history': r'''import concurrent.futures as cf

import pandas as pd
import streamlit as st

from appai import charts
from appai.bench import capacity, summarize_run
from appai.client import AUTH_TYPES, stream_collect
from appai.store import KINDS, display_name
from appai.ui.bench import render_sweep
from appai.ui.chat import format_meta
from appai.ui.state import endpoint_cfg, get_store, missing_key, slo, snapshot

METRICS = [("throughput", "Vazão agregada (tok/s)"), ("ttft_p95", "TTFT p95 (ms)"),
           ("goodput_rps", "Goodput (req/s dentro do SLO)"), ("itl_p95", "ITL p95 (ms)")]


def _delta(v, base, lower_is_better=False):
    if v is None or base in (None, 0):
        return "—"
    d = (v / base - 1) * 100
    good = d < 0 if lower_is_better else d > 0
    return f"{d:+.0f}% {'🟢' if good else '🔴' if d else ''}"


def compare_sweeps(runs: list):
    cur = slo()
    items = []
    for r in runs:
        rows = summarize_run(r["data"]["levels"], cur)
        items.append({"name": r["label"] or r["id"], "mode": r["info"].get("mode", "closed"),
                      "rows": rows, "cap": capacity(rows)})
    if len({i["mode"] for i in items}) > 1:
        st.warning("As execuções usam modos de carga diferentes (closed × open): o eixo de "
                   "carga não é comparável.")
    table = []
    base = None
    for it in items:
        rows = sorted(it["rows"], key=lambda x: x["load"])
        peak = max((x["throughput"] or 0) for x in rows)
        low = rows[0]
        rec = {"Execução": it["name"], "Capacidade no SLO": it["cap"]["max_ok"],
               "Pico (tok/s)": peak, "TTFT p95 na menor carga (ms)": low["ttft_p95"],
               "ITL p95 na menor carga (ms)": low["itl_p95"],
               "Goodput no limite (req/s)": it["cap"]["goodput_rps"]}
        if base is None:
            base = rec
            rec.update({"Δ capacidade": "base", "Δ pico": "base", "Δ TTFT": "base"})
        else:
            rec["Δ capacidade"] = _delta(rec["Capacidade no SLO"], base["Capacidade no SLO"])
            rec["Δ pico"] = _delta(peak, base["Pico (tok/s)"])
            rec["Δ TTFT"] = _delta(rec["TTFT p95 na menor carga (ms)"],
                                   base["TTFT p95 na menor carga (ms)"], lower_is_better=True)
        table.append(rec)
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
    st.caption("Δ em relação à primeira execução selecionada. Capacidade recalculada com o "
               "SLO atual da aba Benchmark.")
    cols = st.columns(2)
    for i, (key, title) in enumerate(METRICS):
        cols[i % 2].plotly_chart(charts.compare_runs(items, key, title), width="stretch",
                                 key=f"cmp-{key}")


def compare_rag(runs: list):
    recs = []
    for r in runs:
        rec = {"Execução": r["label"]}
        for s in r["data"]["summary"]["stages"]:
            rec[f"{s['Etapa']} p50"] = s["p50 (ms)"]
        recs.append(rec)
    st.dataframe(pd.DataFrame(recs), width="stretch", hide_index=True)


def _delete(rid: str):
    # callback: roda antes do multiselect, então pode limpar a seleção
    get_store().delete(rid)
    st.session_state.compare_ids = []


def render_history():
    store = get_store()
    if not store.enabled:
        st.warning(f"Histórico desativado: {store.error}. Defina `APPAI_DATA_DIR` para um "
                   "diretório gravável (no Kubernetes, o PVC montado em /data).")
        return
    runs = store.list()
    if not runs:
        st.info("Nenhuma execução salva ainda. Benchmarks, cold starts e testes RAG são "
                "salvos automaticamente.")
        return
    names = {r["id"]: display_name(r) for r in runs}
    kinds = {r["id"]: r["kind"] for r in runs}
    sel = st.multiselect("Execuções (selecione 2 ou mais do mesmo tipo para comparar)",
                         list(names), format_func=names.get, key="compare_ids")
    if not sel:
        st.dataframe(pd.DataFrame([{"Execução": r["label"], "Tipo": KINDS.get(r["kind"]),
                                    "Modelo": r["model"], "Endpoint": r["endpoint"],
                                    "id": r["id"]} for r in runs]),
                     width="stretch", hide_index=True)
        return
    full = [store.get(i) for i in sel]
    sel_kinds = {kinds[i] for i in sel}
    if len(sel) == 1:
        r = full[0]
        a1, a2, a3 = st.columns([3, 1, 1])
        new = a1.text_input("Rótulo", value=r["label"], key=f"ren-{r['id']}")
        if a2.button("Renomear", width="stretch") and new != r["label"]:
            store.rename(r["id"], new)
            st.rerun()
        a3.button("🗑️ Excluir", width="stretch", on_click=_delete, args=(r["id"],))
        if r["kind"] == "sweep":
            render_sweep(r, key=f"h-{r['id']}")
        elif r["kind"] == "rag":
            st.plotly_chart(charts.rag_stages(r["data"]["summary"]), width="stretch")
            compare_rag(full)
        else:
            st.json({k: v for k, v in r["data"].items() if k != "events"})
        return
    if len(sel_kinds) > 1:
        st.warning("Selecione execuções do mesmo tipo para comparar.")
        return
    kind = sel_kinds.pop()
    if kind == "sweep":
        compare_sweeps(full)
    elif kind == "rag":
        compare_rag(full)
    else:
        st.plotly_chart(charts.coldstart_bars([
            {"name": r["label"], "ready_s": r["data"].get("ready_s"),
             "total_s": r["data"].get("total_s")} for r in full]), width="stretch")


def render_side_by_side():
    s = st.session_state
    st.caption("Mesmo prompt, ao mesmo tempo, em dois endpoints: o A é o da sidebar. "
               "Ex.: NAI on-prem × API de cloud, GPU × GPU, FP16 × quantizado.")
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Endpoint B — URL", key="b_api_url", placeholder="https://.../v1")
        st.text_input("Endpoint B — API Key", key="b_api_key", type="password")
    with c2:
        st.selectbox("Endpoint B — Auth", AUTH_TYPES, key="b_auth_type")
        st.text_input("Endpoint B — Modelo", key="b_model")
    st.text_area("Prompt", key="sbs_prompt", height=80)
    go_btn = st.button("⚖️ Enviar para A e B", type="primary",
                       disabled=not (s.b_api_url.strip() and s.b_model.strip()))
    if go_btn:
        if missing_key():
            st.error("Insira a API Key do endpoint A na sidebar.")
            return
        msgs = [{"role": "user", "content": s.sbs_prompt}]
        cfg_a = snapshot()
        cfg_b = endpoint_cfg(s.b_api_url, s.b_api_key, s.b_auth_type, s.b_model.strip())

        def call(cfg):
            try:
                return stream_collect(cfg, msgs, s.max_tokens)
            except Exception as e:  # noqa: BLE001
                return None, {"error": str(e)[:300]}

        with st.spinner("Aguardando os dois endpoints…"):
            with cf.ThreadPoolExecutor(2) as ex:
                fa, fb = ex.submit(call, cfg_a), ex.submit(call, cfg_b)
                s.sbs_result = {"A": (cfg_a["model"], *fa.result()),
                                "B": (cfg_b["model"], *fb.result())}
    if s.sbs_result:
        cols = st.columns(2)
        for col, side in zip(cols, ("A", "B")):
            model, text, meta = s.sbs_result[side]
            with col:
                st.markdown(f"**{side} · `{model}`**")
                if text is None:
                    st.error(meta["error"])
                else:
                    st.markdown(text)
                    st.caption(format_meta(meta) + f"  ·  total {meta['latency']:.1f} s")


def render():
    t1, t2 = st.tabs(["🗂️ Execuções salvas e comparação", "⚖️ Lado a lado ao vivo"])
    with t1:
        render_history()
    with t2:
        render_side_by_side()
''',
    'appai.ui.rag': r'''import time

import pandas as pd
import streamlit as st

from appai import charts
from appai.client import api_base, build_headers
from appai.rag import (SAMPLE_CORPUS, SAMPLE_QUESTIONS, RagIndex, run_query, run_rag_bench,
                       summarize_rag)
from appai.ui.state import get_store, missing_key, snapshot
from appai.version import build_info

CORPUS_SOURCES = ["Exemplo embutido (infra de IA)", "Meus documentos (.txt / .md)"]


def emb_cfg() -> dict:
    s = st.session_state
    cfg = snapshot()
    if s.rag_emb_url.strip():
        cfg["base"] = api_base(s.rag_emb_url)
    if s.rag_emb_key.strip():
        cfg["headers"] = build_headers(s.rag_emb_key, s.auth_type)
    return cfg


def questions() -> list:
    qs = [q.strip() for q in st.session_state.rag_questions.splitlines() if q.strip()]
    return qs or SAMPLE_QUESTIONS


def render():
    s = st.session_state
    st.caption(
        "Pipeline RAG medido por etapa: **embedding da pergunta → busca vetorial → "
        "prefill → geração**. O TTFT que o usuário sente é a soma das três primeiras, e o "
        "contexto recuperado deixa o prompt bem maior que um chat."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Modelo de embedding", key="rag_emb_model",
                      placeholder="ex.: nai-embed / bge-large / e5-mistral")
        st.text_input("URL de embeddings (vazio = mesmo endpoint)", key="rag_emb_url")
        st.text_input("API Key de embeddings (vazio = mesma)", key="rag_emb_key",
                      type="password")
    with c2:
        st.radio("Corpus", CORPUS_SOURCES, key="rag_corpus_src")
        files = None
        if s.rag_corpus_src.startswith("Meus"):
            files = st.file_uploader("Documentos", type=["txt", "md"],
                                     accept_multiple_files=True)
        st.slider("Palavras por chunk", 40, 400, key="rag_chunk", step=20)
        build = st.button("📚 Indexar corpus", width="stretch",
                          disabled=not s.rag_emb_model.strip())
    with c3:
        st.slider("Chunks recuperados (top-k)", 1, 10, key="rag_k")
        st.slider("Perguntas no teste", 1, 200, key="rag_n")
        st.slider("Concorrência", 1, 32, key="rag_conc")
        st.slider("max_tokens da resposta", 32, 1024, key="rag_max_tokens", step=32)

    if build:
        docs = SAMPLE_CORPUS
        if files:
            docs = [(f.name, f.getvalue().decode("utf-8", errors="replace")) for f in files]
        try:
            idx = RagIndex()
            with st.spinner("Gerando embeddings do corpus…"):
                idx.build(emb_cfg(), s.rag_emb_model.strip(), docs, int(s.rag_chunk))
            s.rag_index = idx
        except Exception as e:
            st.error(f"Falha ao indexar: {e}")

    idx = s.rag_index
    if not idx:
        st.info("Informe o modelo de embedding e clique em **Indexar corpus**. "
                "A busca vetorial roda em memória (numpy) — mede o pipeline e o endpoint "
                "de embeddings, não um banco vetorial de produção.")
        return
    st.success(f"Índice: {idx.stats['chunks']} chunks de {idx.stats['docs']} documento(s), "
               f"dimensão {idx.stats['dim']}, indexado em {idx.stats['seconds']:.1f} s "
               f"({idx.stats['chunks_per_s'] or 0:.0f} chunks/s de embedding).")

    st.text_area("Perguntas (uma por linha; vazio = perguntas de exemplo)", key="rag_questions",
                 height=90, placeholder="\n".join(SAMPLE_QUESTIONS[:3]))
    q1, q2, q3 = st.columns([2, 1, 1])
    q1.text_input("Rótulo da execução", key="rag_label")
    try_one = q2.button("💬 Testar 1 pergunta", width="stretch")
    run = q3.button("▶ Rodar teste RAG", type="primary", width="stretch")

    if (try_one or run) and missing_key():
        st.error("Insira sua API Key na sidebar.")
        return
    if try_one:
        rec = run_query(snapshot(), emb_cfg(), s.rag_emb_model.strip(), idx, questions()[0],
                        int(s.rag_k), int(s.rag_max_tokens), keep_text=True)
        if rec["ok"]:
            st.markdown(f"**{rec['question']}**\n\n{rec['answer']}")
            st.caption(f"Fontes: {', '.join(rec['sources'])} · TTFT percebido "
                       f"{rec['user_ttft_ms']:.0f} ms · total {rec['total_ms']:.0f} ms · "
                       f"{rec['tokens_in']} tokens de entrada")
        else:
            st.error(rec.get("error"))
    if run:
        bar = st.progress(0.0)
        recs = run_rag_bench(snapshot(), emb_cfg(), s.rag_emb_model.strip(), idx, questions(),
                             int(s.rag_n), int(s.rag_conc), int(s.rag_k),
                             int(s.rag_max_tokens),
                             lambda i, n: bar.progress(i / n, text=f"{i}/{n} perguntas"))
        bar.empty()
        summ = summarize_rag(recs)
        b = build_info()
        info = {"endpoint": snapshot()["base"], "model": s.model_name,
                "emb_model": s.rag_emb_model, "k": s.rag_k, "concurrency": s.rag_conc,
                "chunks": idx.stats["chunks"], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "app": f"v{b['version']} ({b['commit']})"}
        label = s.rag_label.strip() or f"{s.model_name} + {s.rag_emb_model} · k={s.rag_k}"
        data = {"summary": summ, "records": recs, "index": idx.stats}
        get_store().save("rag", label, info, data)
        s.rag_result = {"label": label, "info": info, "data": data}

    r = s.rag_result
    if r:
        summ = r["data"]["summary"]
        st.divider()
        by = {x["key"]: x for x in summ["stages"]}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("TTFT percebido p50", f"{by['user_ttft_ms']['p50 (ms)'] or 0:.0f} ms")
        m2.metric("TTFT percebido p95", f"{by['user_ttft_ms']['p95 (ms)'] or 0:.0f} ms")
        m3.metric("Tokens de entrada (média)", f"{summ['tokens_in_mean'] or 0:.0f}")
        m4.metric("Erros", f"{summ['errors']} / {summ['requests']}")
        st.plotly_chart(charts.rag_stages(summ), width="stretch", key="rag-stages")
        st.dataframe(pd.DataFrame(summ["stages"]).drop(columns=["key"]), width="stretch",
                     hide_index=True)
''',
    'appai.ui.sidebar': r'''import streamlit as st

from appai.client import AUTH_TYPES, fetch_models
from appai.store import data_dir
from appai.ui.state import embedded_mock, get_store, snapshot
from appai.version import build_info


def use_mock():
    m = embedded_mock()
    s = st.session_state
    s.api_url, s.auth_type, s.api_key = m["url"] + "/v1", "None / No Auth", ""
    s.model_name = m["sim"].model
    s.models = [m["sim"].model, m["sim"].emb_model]
    s.tele_vllm_url = m["url"] + "/metrics"
    s.tele_prom_url = m["url"]
    s.rag_emb_model = m["sim"].emb_model
    s.cs_hibernate_url = m["url"] + "/admin/hibernate"


def render():
    s = st.session_state
    with st.sidebar:
        st.markdown("### ⚡ Configuração")
        st.text_input("API Endpoint URL", key="api_url")
        st.text_input("API Key", key="api_key", type="password")
        st.selectbox("Auth Header", AUTH_TYPES, key="auth_type",
                     help="NAI normalmente usa Bearer token.")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔌 Conectar", width="stretch"):
                try:
                    s.models = fetch_models(snapshot())
                    st.success(f"{len(s.models)} modelo(s)")
                except Exception as e:
                    st.error(f"Falhou: {e}")
        with col_b:
            if st.button("🗑️ Limpar chat", width="stretch"):
                s.messages = []
                s.debug_log = ""
                st.rerun()

        if s.models:
            if s.model_name not in s.models:
                s.model_name = s.models[0]
            st.selectbox("Modelo", s.models, key="model_name")
        else:
            st.text_input("Modelo / Endpoint Name", key="model_name")

        st.text_area("System Prompt (opcional)", key="system_prompt", height=100,
                     placeholder="You are a helpful assistant…")

        with st.expander("Parâmetros de geração"):
            st.slider("Temperature", 0.0, 2.0, key="temperature", step=0.05)
            st.slider("Max tokens", 64, 8192, key="max_tokens", step=64)
            st.slider("Retries em erro transitório", 1, 8, key="max_retries",
                      help="Inclui o 404 'hibernated endpoint' do NAI.")

        st.toggle("Streaming (SSE)", key="use_stream")
        st.toggle("Verificar TLS", key="verify_tls")
        st.toggle("🐛 Debug", key="show_debug")

        with st.expander("🧪 Mock local (sem GPU)"):
            st.caption("Sobe um servidor que imita vLLM/NAI dentro do próprio app — "
                       "streaming, fila, KV cache, hibernação, embeddings e DCGM "
                       "sintéticos. Para ensaiar a demo; os números não são reais.")
            # on_click roda antes dos widgets: dá para alterar os campos acima.
            st.button("Usar mock embutido", width="stretch", on_click=use_mock)

        st.divider()
        b = build_info()
        store = get_store()
        hist = (f"histórico em `{store.path}`" if store.enabled
                else f"histórico desativado ({data_dir()} não gravável)")
        built = f" · build {b['build_date']}" if b["build_date"] else ""
        st.caption(f"v{b['version']} · commit `{b['commit']}` ({b['source']}){built}  \n{hist}")
''',
    'appai.ui.state': r'''"""Estado da sessão, snapshot de configuração e recursos compartilhados."""

import copy
import os

import streamlit as st

from appai.bench import DEFAULT_SLO
from appai.client import api_base, build_headers
from appai.store import Store
from appai.telemetry import DEFAULT_PROM_QUERIES
from appai.workload import PROFILES

DEFAULTS = {
    # conexão
    "messages": [], "models": [], "debug_log": "",
    "api_url": os.environ.get("NAI_API_URL", "http://10.38.38.223/enterpriseai/v1"),
    "api_key": os.environ.get("NAI_API_KEY", ""),
    "auth_type": "Authorization: Bearer {key}",
    "model_name": os.environ.get("NAI_MODEL", "uep-chat"),
    "system_prompt": "", "temperature": 0.7, "max_tokens": 1024,
    "use_stream": True, "verify_tls": False, "show_debug": False, "max_retries": 4,
    # custo
    "price_preset": "Modelo frontier", "price_in_per_1m": 3.0, "price_out_per_1m": 15.0,
    "gpu_hour": 4.0, "gpu_count": 1, "utilization_pct": 60,
    "monthly_in_m": 1000.0, "monthly_out_m": 500.0,
    "energy_separate": False, "power_w": 400.0, "pue": 1.4, "kwh_price": 0.15,
    "cost_basis": "Vazão dentro do SLO",
    # benchmark
    "bench_mode": "closed", "bench_profile": "Chat curto",
    "bench_in": PROFILES["Chat curto"]["in"], "bench_out": PROFILES["Chat curto"]["out"],
    "bench_dist": "Fixo", "bench_cv": 0.6,
    "bench_levels": [1, 4, 8], "bench_rounds": 3,
    "bench_rates": "0.5, 1, 2, 4", "bench_duration": 20,
    "bench_nocache": True, "bench_force_output": False, "bench_warmup": True,
    "bench_use_dataset": False, "bench_label": "",
    "slo_ttft": DEFAULT_SLO["ttft_ms"], "slo_itl": DEFAULT_SLO["itl_ms"],
    "slo_err": DEFAULT_SLO["max_err_pct"],
    "tele_vllm_url": os.environ.get("VLLM_METRICS_URL", ""), "tele_interval": 1.0,
    "tele_prom_url": os.environ.get("PROMETHEUS_URL", ""), "tele_prom_token": "",
    "tele_prom_queries": DEFAULT_PROM_QUERIES,
    # cold start
    "cs_backend": "Nutanix Files (NFS)", "cs_label": "", "cs_model_gb": 16.0,
    "cs_timeout": 900, "cs_poll": 5.0, "cs_hibernate_url": "",
    # rag
    "rag_emb_model": os.environ.get("NAI_EMBED_MODEL", ""), "rag_emb_url": "",
    "rag_emb_key": "", "rag_k": 4, "rag_n": 20, "rag_conc": 4, "rag_max_tokens": 192,
    "rag_chunk": 120, "rag_corpus_src": "Exemplo embutido (infra de IA)",
    "rag_questions": "", "rag_label": "",
    # lado a lado
    "b_api_url": "", "b_api_key": "", "b_auth_type": "Authorization: Bearer {key}",
    "b_model": "", "sbs_prompt": "Explique em 3 frases o que é KV cache.",
}

# Estado que não é widget (resultados) — não entra no "pin" abaixo.
RESULTS = {"bench_result": None, "bench_dataset": [], "bench_dataset_name": "",
           "cs_result": None, "rag_index": None, "rag_result": None, "sbs_result": None,
           "compare_ids": []}


def init_state():
    for k, v in {**DEFAULTS, **RESULTS}.items():
        if k not in st.session_state:
            # cópia: listas/dicts dos defaults não podem ser compartilhados entre
            # sessões (o módulo é importado uma vez por processo).
            st.session_state[k] = copy.deepcopy(v)
    # Streamlit apaga o estado de widgets que não foram renderizados nesta
    # execução (ex.: campos do modo closed quando o modo open está ativo).
    # Reatribuir mantém os valores escolhidos pelo usuário.
    for k in DEFAULTS:
        st.session_state[k] = st.session_state[k]


def slo() -> dict:
    s = st.session_state
    return {"ttft_ms": float(s.slo_ttft), "itl_ms": float(s.slo_itl),
            "max_err_pct": float(s.slo_err)}


def endpoint_cfg(url: str, key: str, auth: str, model: str, extra: dict | None = None) -> dict:
    s = st.session_state
    return {
        "base": api_base(url), "headers": build_headers(key, auth), "verify": s.verify_tls,
        "model": model, "temperature": s.temperature, "max_retries": int(s.max_retries),
        "prices": (float(s.price_in_per_1m), float(s.price_out_per_1m)),
        "extra": extra or {},
    }


def snapshot(extra: dict | None = None) -> dict:
    """Foto da configuração A (sidebar). As threads do benchmark não podem
    acessar st.session_state, por isso tudo vai num dict."""
    s = st.session_state
    return endpoint_cfg(s.api_url, s.api_key, s.auth_type, s.model_name, extra)


def missing_key() -> bool:
    s = st.session_state
    return not s.auth_type.startswith("None") and not s.api_key.strip()


@st.cache_resource
def get_store() -> Store:
    return Store()


@st.cache_resource
def embedded_mock(port: int = 8765):
    """Mock vLLM/NAI no mesmo processo, para ensaio de demo sem GPU."""
    from appai.mock_server import serve
    srv, sim = serve(port, decode_tps=45, max_batch=16, kv_tokens=40000, wake_s=15)
    return {"url": f"http://127.0.0.1:{port}", "sim": sim, "server": srv}
''',
    'appai.version': r'''"""Versão e commit do build, exibidos na sidebar e gravados em cada execução.

Três formas de rodar, três origens:
- container: build-args do CI (APP_VERSION, GIT_SHA, BUILD_DATE);
- arquivo único (app_standalone.py): hash do código-fonte embutido no bundle;
- clone local: `git rev-parse`.
"""

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

VERSION = "2.0.0"


def _bundle() -> dict | None:
    """Metadados que o app_standalone.py coloca no pacote ao carregá-lo."""
    return getattr(sys.modules.get("appai"), "__bundle__", None)


def _git_sha() -> str:
    here = globals().get("__file__")
    if not here:
        return ""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(here).resolve().parent.parent,
            capture_output=True, text=True, timeout=2, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


@lru_cache(maxsize=1)
def build_info() -> dict:
    sha = os.environ.get("GIT_SHA", "").strip()
    date = os.environ.get("BUILD_DATE", "").strip()
    bundle = _bundle()
    if sha:
        source, commit = "container", sha[:7]
    elif bundle:
        source, commit = "arquivo único", f"src {bundle['digest']}"
    else:
        sha = _git_sha()
        source, commit = "local", sha[:7] if sha else "desconhecido"
    return {
        "version": os.environ.get("APP_VERSION", "").strip() or VERSION,
        "commit": commit,
        "build_date": date,
        "source": source,
    }
''',
    'appai.workload': r'''"""Geração de carga: prompts sintéticos, distribuição de tamanhos e datasets.

Contagem de tokens é aproximada (~0,75 palavra por token, bom para PT/EN);
o número real vem do `usage` devolvido pelo servidor.
"""

import io
import json
import math
import random
import uuid

PROFILES = {
    "Chat curto": {
        "in": 50, "out": 128,
        "q": "Explique em poucas frases o que é Kubernetes.",
    },
    "RAG / prompt longo": {
        "in": 2000, "out": 256,
        "q": "Com base no texto acima, resuma os pontos principais.",
    },
    "Geração longa": {
        "in": 100, "out": 1024,
        "q": "Escreva um texto detalhado sobre boas práticas de infraestrutura para IA.",
    },
    "Personalizado": {
        "in": 200, "out": 256,
        "q": "Resuma o texto acima em tópicos.",
    },
}

FILLER = (
    "A plataforma de infraestrutura para inteligência artificial precisa equilibrar "
    "latência, vazão e custo. GPUs, redes de alta velocidade e armazenamento de baixa "
    "latência determinam o desempenho da inferência em produção. "
)

DISTRIBUTIONS = ["Fixo", "Lognormal (realista)"]
WORDS_PER_TOKEN = 0.75
MAX_TOKENS_CAP = 32768


def approx_tokens(text: str) -> int:
    return max(1, round(len(text.split()) / WORDS_PER_TOKEN))


def build_prompt(question: str, target_tokens: int, nocache: bool,
                 rng: random.Random | None = None) -> str:
    """Monta um prompt com ~target_tokens tokens de contexto + a pergunta.
    Com nocache, um prefixo único por requisição evita prefix caching no servidor."""
    words = FILLER.split()
    n_words = max(0, int(target_tokens * WORDS_PER_TOKEN) - len(question.split()))
    reps, rest = divmod(n_words, len(words))
    context = FILLER * reps + " ".join(words[:rest])
    tag = (rng.getrandbits(32) if rng else uuid.uuid4().int & 0xFFFFFFFF)
    prefix = f"[ref {tag:08x}]\n" if nocache else ""
    return f"{prefix}{context}\n\n{question}".strip()


def lognormal(mean: float, cv: float, rng: random.Random) -> float:
    """Amostra lognormal com média `mean` e coeficiente de variação `cv`."""
    if cv <= 0:
        return mean
    sigma2 = math.log(1 + cv * cv)
    mu = math.log(mean) - sigma2 / 2
    return rng.lognormvariate(mu, math.sqrt(sigma2))


def sample_size(mean: int, dist: str, cv: float, rng: random.Random) -> int:
    if dist.startswith("Lognormal"):
        v = lognormal(max(mean, 1), cv, rng)
        return int(min(max(round(v), 8), max(mean * 6, 16), MAX_TOKENS_CAP))
    return int(mean)


def make_requests(n: int, spec: dict, rng: random.Random) -> list:
    """Gera n requisições {prompt, max_tokens}.

    spec: question, in, out, dist, cv, nocache, dataset (lista opcional de
    {prompt, out}). Com dataset, amostra prompts reais em vez dos sintéticos.
    """
    dataset = spec.get("dataset") or []
    reqs = []
    for _ in range(n):
        if dataset:
            item = rng.choice(dataset)
            prompt = item["prompt"]
            if spec.get("nocache"):
                prompt = f"[ref {rng.getrandbits(32):08x}]\n{prompt}"
            out = item.get("out") or spec["out"]
            reqs.append({"prompt": prompt, "max_tokens": int(min(out, MAX_TOKENS_CAP))})
            continue
        tin = sample_size(spec["in"], spec.get("dist", "Fixo"), spec.get("cv", 0.6), rng)
        tout = sample_size(spec["out"], spec.get("dist", "Fixo"), spec.get("cv", 0.6), rng)
        reqs.append({
            "prompt": build_prompt(spec["question"], tin, spec.get("nocache", True), rng),
            "max_tokens": tout,
        })
    return reqs


def _sharegpt_pairs(items: list) -> list:
    out = []
    for it in items:
        conv = it.get("conversations") or it.get("conversation") or []
        human = next((c for c in conv if c.get("from") in ("human", "user")), None)
        gpt = next((c for c in conv if c.get("from") in ("gpt", "assistant")), None)
        if human and human.get("value"):
            o = approx_tokens(gpt["value"]) if gpt and gpt.get("value") else None
            out.append({"prompt": human["value"], "out": o})
    return out


def load_dataset(data: bytes, filename: str, limit: int = 5000) -> list:
    """Carrega prompts reais. Aceita:
    - JSONL com campo "prompt" (e opcional "max_tokens" / "output_tokens");
    - JSON/JSONL no formato ShareGPT ("conversations": [{from, value}, ...]);
    - TXT com um prompt por linha.
    Devolve [{prompt, out}] (out = tokens de saída esperados ou None)."""
    text = data.decode("utf-8", errors="replace")
    name = filename.lower()
    items: list = []
    if name.endswith(".json"):
        parsed = json.loads(text)
        items = parsed if isinstance(parsed, list) else [parsed]
    elif name.endswith(".jsonl"):
        for line in io.StringIO(text):
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    else:
        return [{"prompt": ln.strip(), "out": None}
                for ln in text.splitlines() if ln.strip()][:limit]

    if items and isinstance(items[0], dict) and ("conversations" in items[0]
                                                 or "conversation" in items[0]):
        return _sharegpt_pairs(items)[:limit]
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        prompt = it.get("prompt") or it.get("input") or it.get("question")
        if prompt:
            o = it.get("max_tokens") or it.get("output_tokens") or it.get("output_len")
            out.append({"prompt": str(prompt), "out": int(o) if o else None})
    return out[:limit]


def dataset_stats(dataset: list) -> dict:
    if not dataset:
        return {}
    ins = sorted(approx_tokens(d["prompt"]) for d in dataset)
    outs = sorted(d["out"] for d in dataset if d.get("out"))
    mid = lambda v: v[len(v) // 2] if v else None  # noqa: E731
    return {"n": len(dataset), "in_median": mid(ins), "in_max": ins[-1],
            "out_median": mid(outs), "out_known": len(outs)}
'''
}
_PACKAGES = {'appai', 'appai.ui'}
_DIGEST = "ad1049b4c5"


class _AppaiBundleImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Entrega os módulos appai.* a partir de _SOURCES."""

    _appai_bundle = True

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _SOURCES:
            return None
        return importlib.util.spec_from_loader(fullname, self,
                                               is_package=fullname in _PACKAGES)

    def create_module(self, spec):
        return None

    def get_source(self, fullname):
        return _SOURCES[fullname]

    def exec_module(self, module):
        name = module.__name__
        src = _SOURCES[name]
        # nome "virtual" só para tracebacks legíveis (linecache tem o código)
        fn = "<app_standalone>/" + name.replace(".", "/") + (
            "/__init__.py" if name in _PACKAGES else ".py")
        linecache.cache[fn] = (len(src), None, src.splitlines(True), fn)
        if name == "appai":
            module.__bundle__ = {"digest": _DIGEST}
        exec(compile(src, fn, "exec"), module.__dict__)


# O Streamlit reexecuta este arquivo a cada interação: instala uma vez só.
if not any(getattr(f, "_appai_bundle", False) for f in sys.meta_path):
    sys.meta_path.insert(0, _AppaiBundleImporter())


def _inside_streamlit() -> bool:
    try:
        from streamlit import runtime
        return runtime.exists()
    except Exception:
        return False


if __name__ == "__main__" and sys.argv[1:2] == ["mock"] and not _inside_streamlit():
    from appai.mock_server import main as _mock_main
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    _mock_main()
    sys.exit(0)

# ── app.py ────────────────────────────────────────────────────────────────────
import streamlit as st
import urllib3

st.set_page_config(page_title="AI Infra Benchmark", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

from appai.ui import bench, chat, coldstart, cost, history, rag, sidebar  # noqa: E402
from appai.ui.state import init_state, snapshot  # noqa: E402

init_state()
if not st.session_state.verify_tls:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

sidebar.render()

st.markdown("## ⚡ AI Infra Benchmark")
st.caption(f"Endpoint `{snapshot()['base']}`  ·  modelo `{st.session_state.model_name}`")
kpi_slot = st.empty()
chat.render_kpis(kpi_slot)

tabs = st.tabs(["💬 Chat", "📊 Benchmark", "🧊 Cold start", "📚 RAG",
                "🗂️ Histórico & A/B", "💰 Custo"])
with tabs[0]:
    chat.render(kpi_slot)
with tabs[1]:
    bench.render()
with tabs[2]:
    coldstart.render()
with tabs[3]:
    rag.render()
with tabs[4]:
    history.render()
with tabs[5]:
    cost.render()

"""Figuras Plotly compartilhadas entre a UI e o relatório HTML."""

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

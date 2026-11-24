"""Relatório HTML autocontido de uma execução de benchmark, para enviar ao
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

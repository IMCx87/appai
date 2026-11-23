import json
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

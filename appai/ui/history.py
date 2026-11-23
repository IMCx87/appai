import concurrent.futures as cf

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

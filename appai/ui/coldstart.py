import time

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

"""
AI Infra Benchmark — NAI Endpoint Chat
Cliente de chat + benchmark para endpoints OpenAI-compatible
(vLLM / Nutanix Enterprise AI).

Executar:  pip install -r requirements.txt && streamlit run app.py
Tema:      opcional — .streamlit/config.toml aplica o dark theme.
Env vars:  NAI_API_URL, NAI_API_KEY, NAI_MODEL, NAI_EMBED_MODEL,
           VLLM_METRICS_URL, PROMETHEUS_URL, APPAI_DATA_DIR.
Sem GPU:   "Mock local" na sidebar, ou `python -m appai.mock_server`.
"""

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

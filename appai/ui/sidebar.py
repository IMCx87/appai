import streamlit as st

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

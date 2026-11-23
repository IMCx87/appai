"""Estado da sessão, snapshot de configuração e recursos compartilhados."""

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

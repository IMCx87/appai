"""
NAI Endpoint Chat
Demo client para endpoints OpenAI-compatible (vLLM / Nutanix Enterprise AI).

Executar:  streamlit run nai_chat_app.py
Tema:      coloque o config.toml em .streamlit/config.toml ao lado deste arquivo.
"""

import json
import random
import time

import requests
import streamlit as st
import urllib3

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NAI Endpoint Chat",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estado da sessão ──────────────────────────────────────────────────────────
DEFAULTS = {
    "messages": [],
    "models": [],
    "debug_log": "",
    "api_url": "http://10.38.38.223/enterpriseai/v1",
    "api_key": "",
    "auth_type": "Authorization: Bearer {key}",
    "model_name": "uep-chat",
    "system_prompt": "",
    "temperature": 0.7,
    "max_tokens": 1024,
    "use_stream": True,
    "verify_tls": False,
    "show_debug": False,
    "max_retries": 4,
}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

if not st.session_state.verify_tls:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ── Helpers ───────────────────────────────────────────────────────────────────
def api_base() -> str:
    base = st.session_state.api_url.rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def build_headers() -> dict:
    h = {"Content-Type": "application/json"}
    key = st.session_state.api_key.strip()
    auth = st.session_state.auth_type
    if auth.startswith("Authorization"):
        h["Authorization"] = f"Bearer {key}"
    elif auth.startswith("x-api-key"):
        h["x-api-key"] = key
    return h


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
    merged = [filtered[0]]
    for msg in filtered[1:]:
        if msg["role"] == merged[-1]["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(msg)
    while merged and merged[0]["role"] != "user":
        merged.pop(0)
    return merged


def build_body(stream: bool) -> dict:
    body_messages = []
    if st.session_state.system_prompt.strip():
        body_messages.append(
            {"role": "system", "content": st.session_state.system_prompt.strip()}
        )
    body_messages.extend(sanitize_messages(st.session_state.messages))
    body = {
        "model": st.session_state.model_name,
        "messages": body_messages,
        "max_tokens": st.session_state.max_tokens,
        "temperature": st.session_state.temperature,
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def log_request(endpoint: str, headers: dict, body: dict):
    log = f"POST {endpoint}\n"
    log += f"Headers: {json.dumps(redact(headers), indent=2)}\n"
    log += f"Body: {json.dumps(body, indent=2, ensure_ascii=False)}\n"
    st.session_state.debug_log = log


def fetch_models() -> list:
    r = requests.get(
        api_base() + "/models",
        headers=build_headers(),
        timeout=10,
        verify=st.session_state.verify_tls,
    )
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


def post_with_retry(endpoint: str, headers: dict, body: dict,
                    stream: bool, notify=None):
    """POST com retry exponencial + jitter. Retorna Response com status < 400.
    notify(msg) é chamado a cada nova tentativa, para feedback na UI."""
    max_retries = int(st.session_state.max_retries)
    resp = None
    for attempt in range(1, max_retries + 1):
        resp = requests.post(
            endpoint, headers=headers, json=body, stream=stream,
            timeout=(10, 300), verify=st.session_state.verify_tls,
        )
        if resp.status_code < 400:
            return resp
        err_body = resp.text[:1500]
        st.session_state.debug_log += (
            f"\nTentativa {attempt}/{max_retries}: HTTP {resp.status_code}\n{err_body}\n"
        )
        if attempt < max_retries and is_retryable(resp.status_code, err_body):
            wait = min(2 ** attempt, 10) + random.uniform(0, 0.5)
            if notify:
                hib = "endpoint hibernado no pool" if "hibernat" in err_body.lower() \
                      else f"HTTP {resp.status_code}"
                notify(f"⏳ {hib}. Nova tentativa {attempt + 1}/{max_retries} "
                       f"em {wait:.1f}s…")
            resp.close()
            time.sleep(wait)
            continue
        resp.raise_for_status()
    resp.raise_for_status()


def stream_response(resp, metrics: dict):
    """Generator SSE: entrega texto token a token e mede TTFT / throughput."""
    metrics.update(
        {"start": metrics.get("start", time.perf_counter()), "ttft": None,
         "chunks": 0, "usage": None, "model": None, "end": None}
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
                    if metrics["ttft"] is None:
                        metrics["ttft"] = time.perf_counter() - metrics["start"]
                    metrics["chunks"] += 1
                    yield delta
    metrics["end"] = time.perf_counter()


def finalize_metrics(metrics: dict) -> dict:
    meta = {"model": metrics.get("model") or st.session_state.model_name}
    usage = metrics.get("usage") or {}
    out_tokens = usage.get("completion_tokens") or metrics.get("chunks") or 0
    meta["in"] = usage.get("prompt_tokens", 0)
    meta["out"] = out_tokens
    ttft = metrics.get("ttft")
    end, start = metrics.get("end"), metrics.get("start")
    if ttft is not None:
        meta["ttft"] = ttft
    if end and start and ttft is not None and out_tokens > 1:
        gen_time = max(end - start - ttft, 1e-6)
        meta["tps"] = out_tokens / gen_time
    return meta


def format_meta(meta: dict) -> str:
    parts = []
    if meta.get("tps"):
        parts.append(f"⚡ {meta['tps']:.1f} tok/s")
    if meta.get("ttft") is not None:
        parts.append(f"TTFT {meta['ttft'] * 1000:.0f} ms")
    if meta.get("out"):
        parts.append(f"{meta['in']} in / {meta['out']} out tokens")
    if meta.get("model"):
        parts.append(meta["model"])
    return "  ·  ".join(parts)


def call_blocking(endpoint: str, headers: dict, body: dict, notify=None) -> tuple:
    start = time.perf_counter()
    resp = post_with_retry(endpoint, headers, body, stream=False, notify=notify)
    st.session_state.debug_log += f"\nStatus: {resp.status_code}\nResponse: {resp.text[:2000]}\n"
    data = resp.json()
    elapsed = time.perf_counter() - start
    usage = data.get("usage", {})
    content = data["choices"][0]["message"]["content"]
    out = usage.get("completion_tokens", 0)
    meta = {
        "model": data.get("model", st.session_state.model_name),
        "in": usage.get("prompt_tokens", 0),
        "out": out,
    }
    if out > 0 and elapsed > 0:
        meta["tps"] = out / elapsed
    return content, meta


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Configuração")

    st.text_input("API Endpoint URL", key="api_url")
    st.text_input("API Key", key="api_key", type="password")
    st.selectbox(
        "Auth Header",
        ["Authorization: Bearer {key}", "x-api-key: {key}", "None / No Auth"],
        key="auth_type",
        help="NAI normalmente usa Bearer token.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔌 Conectar", use_container_width=True):
            try:
                st.session_state.models = fetch_models()
                st.success(f"{len(st.session_state.models)} modelo(s)")
            except Exception as e:
                st.error(f"Falhou: {e}")
    with col_b:
        if st.button("🗑️ Limpar", use_container_width=True):
            st.session_state.messages = []
            st.session_state.debug_log = ""
            st.rerun()

    if st.session_state.models:
        if st.session_state.model_name not in st.session_state.models:
            st.session_state.model_name = st.session_state.models[0]
        st.selectbox("Modelo", st.session_state.models, key="model_name")
    else:
        st.text_input("Modelo / Endpoint Name", key="model_name")

    st.text_area(
        "System Prompt (opcional)",
        key="system_prompt",
        height=100,
        placeholder="You are a helpful assistant…",
    )

    with st.expander("Parâmetros de geração"):
        st.slider("Temperature", 0.0, 2.0, key="temperature", step=0.05)
        st.slider("Max tokens", 64, 8192, key="max_tokens", step=64)
        st.slider("Retries em erro transitório", 1, 8, key="max_retries",
                  help="Inclui o 404 'hibernated endpoint' do NAI.")

    st.toggle("Streaming (SSE)", key="use_stream")
    st.toggle("Verificar TLS", key="verify_tls")
    st.toggle("🐛 Debug", key="show_debug")


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("## 💬 AI Endpoint Chat")
st.caption(f"Conectado a `{api_base()}`  ·  modelo `{st.session_state.model_name}`")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.caption(format_meta(msg["meta"]))

if st.session_state.show_debug and st.session_state.debug_log:
    with st.expander("🐛 Debug, última requisição", expanded=False):
        st.code(st.session_state.debug_log, language="http")

prompt = st.chat_input("Digite sua mensagem…")

if prompt:
    needs_key = not st.session_state.auth_type.startswith("None")
    if needs_key and not st.session_state.api_key.strip():
        st.error("Insira sua API Key na sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    endpoint = api_base() + "/chat/completions"
    headers = build_headers()
    body = build_body(stream=st.session_state.use_stream)
    log_request(endpoint, headers, body)

    with st.chat_message("assistant", avatar="🤖"):
        notice = st.empty()

        def notify(msg: str):
            notice.info(msg)

        try:
            if st.session_state.use_stream:
                metrics: dict = {"start": time.perf_counter()}
                resp = post_with_retry(endpoint, headers, body,
                                       stream=True, notify=notify)
                notice.empty()
                content = st.write_stream(stream_response(resp, metrics))
                meta = finalize_metrics(metrics)
                st.session_state.debug_log += (
                    f"\nStatus: 200 (stream)\nUsage: {json.dumps(metrics.get('usage'))}\n"
                )
            else:
                with st.spinner("Aguardando resposta…"):
                    content, meta = call_blocking(endpoint, headers, body,
                                                  notify=notify)
                notice.empty()
                st.markdown(content)

            st.caption(format_meta(meta))
            st.session_state.messages.append(
                {"role": "assistant", "content": content, "meta": meta}
            )
        except requests.exceptions.HTTPError as e:
            notice.empty()
            st.error(f"Erro HTTP {e.response.status_code} após "
                     f"{st.session_state.max_retries} tentativa(s): "
                     f"{e.response.text[:500]}")
        except requests.exceptions.ConnectionError as e:
            notice.empty()
            st.error(f"Endpoint inacessível: {e}")
        except Exception as e:
            notice.empty()
            st.error(f"Erro: {e}")

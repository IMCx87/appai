"""
AI Infra Benchmark — NAI Endpoint Chat
Cliente de chat + benchmark para endpoints OpenAI-compatible
(vLLM / Nutanix Enterprise AI).

Executar:  pip install -r requirements.txt && streamlit run app.py
Tema:      opcional — .streamlit/config.toml ao lado deste arquivo aplica o dark theme.
Env vars:  NAI_API_URL e NAI_API_KEY definem os valores iniciais da sidebar.
"""

import concurrent.futures as cf
import json
import os
import random
import statistics
import time
import uuid

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import urllib3
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Infra Benchmark",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Referências ───────────────────────────────────────────────────────────────
# Preços ILUSTRATIVOS por 1M tokens (entrada, saída) em US$ — ajuste ao provedor.
CLOUD_PRESETS = {
    "Modelo pequeno (~8B)": (0.15, 0.60),
    "Modelo médio (~70B)": (0.90, 0.90),
    "Modelo frontier": (3.00, 15.00),
    "Personalizado": None,
}

# Perfis de carga do benchmark: tokens de entrada (aprox.), max_tokens de saída.
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

# ── Estado da sessão ──────────────────────────────────────────────────────────
DEFAULTS = {
    "messages": [],
    "models": [],
    "debug_log": "",
    "api_url": os.environ.get("NAI_API_URL", "http://10.38.38.223/enterpriseai/v1"),
    "api_key": os.environ.get("NAI_API_KEY", ""),
    "auth_type": "Authorization: Bearer {key}",
    "model_name": "uep-chat",
    "system_prompt": "",
    "temperature": 0.7,
    "max_tokens": 1024,
    "use_stream": True,
    "verify_tls": False,
    "show_debug": False,
    "max_retries": 4,
    # custo cloud
    "price_preset": "Modelo frontier",
    "price_in_per_1m": 3.0,
    "price_out_per_1m": 15.0,
    # custo on-prem (premissas ilustrativas)
    "gpu_hour": 4.0,
    "gpu_count": 1,
    "monthly_out_tokens_m": 500.0,
    # benchmark
    "bench_profile": "Chat curto",
    "bench_in": PROFILES["Chat curto"]["in"],
    "bench_out": PROFILES["Chat curto"]["out"],
    "bench_levels": [1, 4, 8],
    "bench_rounds": 3,
    "bench_nocache": True,
    "bench_rows": [],
    "bench_raw": [],
    "bench_info": {},
}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

if not st.session_state.verify_tls:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ── Helpers de configuração ───────────────────────────────────────────────────
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


def snapshot() -> dict:
    """Foto da configuração para uso fora do script principal: as threads do
    benchmark não podem acessar st.session_state."""
    s = st.session_state
    return {
        "base": api_base(),
        "headers": build_headers(),
        "verify": s.verify_tls,
        "model": s.model_name,
        "temperature": s.temperature,
        "max_retries": int(s.max_retries),
        "prices": (float(s.price_in_per_1m), float(s.price_out_per_1m)),
    }


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


def estimate_cost(in_tokens: int, out_tokens: int, prices: tuple) -> float:
    """Custo estimado se a mesma troca rodasse em uma cloud pública
    (preços em US$ por 1M de tokens: entrada, saída)."""
    return in_tokens / 1_000_000 * prices[0] + out_tokens / 1_000_000 * prices[1]


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


def post_with_retry(cfg: dict, body: dict, stream: bool, notify=None, log=None):
    """POST com retry exponencial + jitter. Retorna (Response, nº de retries).
    notify(msg) é chamado a cada nova tentativa; log (lista) recebe o detalhe
    de cada falha. Não usa st.session_state, então roda em threads."""
    endpoint = cfg["base"] + "/chat/completions"
    max_retries = cfg["max_retries"]
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
    meta["in"] = usage.get("prompt_tokens", 0)
    meta["out"] = out_tokens
    ttft = metrics.get("ttft")
    end, start = metrics.get("end"), metrics.get("start")
    if ttft is not None:
        meta["ttft"] = ttft
    if end and start and ttft is not None and out_tokens > 1:
        gen_time = max(end - start - ttft, 1e-6)
        meta["tps"] = out_tokens / gen_time
    t_first, t_last, chunks = metrics.get("t_first"), metrics.get("t_last"), metrics.get("chunks", 0)
    if t_first and t_last and chunks > 1:
        meta["itl"] = (t_last - t_first) / (chunks - 1) * 1000  # ms entre tokens
    meta["cost"] = estimate_cost(meta["in"], meta["out"], prices)
    return meta


def fmt_cost(cost: float) -> str:
    return f"US$ {cost:.5f}" if cost < 0.01 else f"US$ {cost:.4f}"


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


# ── Benchmark ─────────────────────────────────────────────────────────────────
def build_prompt(profile: str, target_tokens: int, nocache: bool) -> str:
    """Monta um prompt com ~target_tokens tokens de contexto + a pergunta do perfil.
    Com nocache, um prefixo único por requisição evita prefix caching no servidor."""
    words = FILLER.split()
    reps = max(0, int(target_tokens * 0.75) // len(words))
    prefix = f"[ref {uuid.uuid4().hex[:8]}]\n" if nocache else ""
    return f"{prefix}{FILLER * reps}\n\n{PROFILES[profile]['q']}".strip()


def run_one(cfg: dict, prompt: str, max_tokens: int) -> dict:
    """Uma requisição de benchmark (streaming). Nunca lança: erros viram registro."""
    body = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": cfg["temperature"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    metrics = {"start": time.perf_counter()}
    retries = 0
    try:
        resp, retries = post_with_retry(cfg, body, stream=True)
        for _ in stream_response(resp, metrics):
            pass
        meta = finalize_metrics(metrics, cfg["model"], cfg["prices"])
        meta.update(ok=True, retries=retries,
                    latency=metrics["end"] - metrics["start"])
        return meta
    except Exception as e:  # noqa: BLE001 — qualquer falha é um dado do benchmark
        return {"ok": False, "error": str(e)[:200], "retries": retries,
                "latency": time.perf_counter() - metrics["start"]}


def pct(values: list, p: float):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def summarize(level: int, records: list, wall: float) -> dict:
    ok = [r for r in records if r["ok"]]
    ttft = [r["ttft"] * 1000 for r in ok if r.get("ttft") is not None]
    itl = [r["itl"] for r in ok if r.get("itl")]
    tps = [r["tps"] for r in ok if r.get("tps")]
    out_tok = sum(r["out"] for r in ok)
    r2 = lambda x: None if x is None else round(x, 1)  # noqa: E731
    return {
        "Concorrência": level,
        "Requisições": len(records),
        "Erros": len(records) - len(ok),
        "Retries": sum(r.get("retries", 0) for r in records),
        "TTFT p50 (ms)": r2(pct(ttft, 50)),
        "TTFT p95 (ms)": r2(pct(ttft, 95)),
        "TTFT p99 (ms)": r2(pct(ttft, 99)),
        "ITL médio (ms)": r2(statistics.fmean(itl)) if itl else None,
        "tok/s por req": r2(statistics.fmean(tps)) if tps else None,
        "Vazão agregada (tok/s)": r2(out_tok / wall) if wall > 0 else None,
        "Req/s": round(len(ok) / wall, 2) if wall > 0 else None,
        "Tokens in": sum(r.get("in", 0) for r in ok),
        "Tokens out": out_tok,
        "Custo cloud (US$)": round(sum(r.get("cost", 0.0) for r in ok), 5),
        "Duração (s)": round(wall, 1),
    }


def run_level(cfg: dict, level: int, prompts: list, max_tokens: int, on_progress):
    """Dispara len(prompts) requisições com `level` workers em paralelo."""
    records, t0 = [], time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=level) as ex:
        futs = [ex.submit(run_one, cfg, p, max_tokens) for p in prompts]
        for i, f in enumerate(cf.as_completed(futs), 1):
            records.append(f.result())
            on_progress(i, len(prompts))
    return records, time.perf_counter() - t0


def apply_profile():
    p = PROFILES[st.session_state.bench_profile]
    st.session_state.bench_in, st.session_state.bench_out = p["in"], p["out"]


def apply_cloud_preset():
    preset = CLOUD_PRESETS.get(st.session_state.price_preset)
    if preset:
        st.session_state.price_in_per_1m, st.session_state.price_out_per_1m = preset


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
        if st.button("🔌 Conectar", width="stretch"):
            try:
                st.session_state.models = fetch_models()
                st.success(f"{len(st.session_state.models)} modelo(s)")
            except Exception as e:
                st.error(f"Falhou: {e}")
    with col_b:
        if st.button("🗑️ Limpar", width="stretch"):
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


# ── Header + KPIs da sessão ───────────────────────────────────────────────────
st.markdown("## ⚡ AI Infra Benchmark")
st.caption(f"Endpoint `{api_base()}`  ·  modelo `{st.session_state.model_name}`")

kpi_slot = st.empty()


def render_kpis():
    """Faixa de KPIs da sessão. Chamada de novo após cada resposta do chat para
    não ficar uma rodada defasada."""
    k = session_kpis()
    with kpi_slot.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Respostas", k["n"])
        c2.metric("TTFT médio", f"{k['ttft']:.0f} ms" if k["ttft"] is not None else "—")
        c3.metric("Tokens/s médio", f"{k['tps']:.1f}" if k["tps"] is not None else "—")
        c4.metric("Tokens in / out", f"{k['tin']} / {k['tout']}")
        c5.metric("Custo cloud acumulado", fmt_cost(k["cost"]) if k["n"] else "—")


render_kpis()

tab_chat, tab_bench, tab_cost = st.tabs(["💬 Chat", "📊 Benchmark", "💰 Custo"])

# ── Aba Chat ──────────────────────────────────────────────────────────────────
with tab_chat:
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

        cfg = snapshot()
        endpoint = cfg["base"] + "/chat/completions"
        body = build_body(stream=st.session_state.use_stream)
        log_request(endpoint, cfg["headers"], body)
        req_log: list = []

        with st.chat_message("assistant", avatar="🤖"):
            notice = st.empty()

            def notify(msg: str):
                notice.info(msg)

            try:
                if st.session_state.use_stream:
                    metrics: dict = {"start": time.perf_counter()}
                    resp, _ = post_with_retry(cfg, body, stream=True,
                                              notify=notify, log=req_log)
                    notice.empty()
                    content = st.write_stream(stream_response(resp, metrics))
                    meta = finalize_metrics(metrics, cfg["model"], cfg["prices"])
                    req_log.append(
                        f"Status: 200 (stream)\nUsage: {json.dumps(metrics.get('usage'))}"
                    )
                else:
                    with st.spinner("Aguardando resposta…"):
                        content, meta = call_blocking(cfg, body, notify=notify,
                                                      log=req_log)
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
            finally:
                if req_log:
                    st.session_state.debug_log += "\n" + "\n".join(req_log) + "\n"
        render_kpis()

# ── Aba Benchmark ─────────────────────────────────────────────────────────────
with tab_bench:
    st.caption(
        "Dispara requisições em streaming com concorrência crescente e mede o "
        "que importa para dimensionar infraestrutura de inferência: TTFT (p50/p95/p99), "
        "latência entre tokens, tokens/s e vazão agregada."
    )
    b1, b2, b3 = st.columns([2, 2, 2])
    with b1:
        st.selectbox("Perfil de carga", list(PROFILES), key="bench_profile",
                     on_change=apply_profile)
        st.slider("Tokens de entrada (aprox.)", 16, 8000, key="bench_in", step=16)
        st.slider("Tokens de saída (max_tokens)", 16, 4096, key="bench_out", step=16)
    with b2:
        st.multiselect("Níveis de concorrência", [1, 2, 4, 8, 16, 32, 64],
                       key="bench_levels")
        st.slider("Rodadas por nível", 1, 10, key="bench_rounds",
                  help="Requisições por worker: total do nível = concorrência × rodadas.")
        st.toggle("Evitar prefix cache (prefixo único por requisição)",
                  key="bench_nocache")
    levels = sorted(st.session_state.bench_levels)
    total_reqs = sum(lv * st.session_state.bench_rounds for lv in levels)
    with b3:
        st.metric("Requisições nesta execução", total_reqs)
        st.caption("⚠️ O benchmark consome tokens/GPU reais do endpoint.")
        run = st.button("▶ Executar benchmark", type="primary",
                        width="stretch", key="run_bench",
                        disabled=not levels)

    if run:
        if not st.session_state.auth_type.startswith("None") \
                and not st.session_state.api_key.strip():
            st.error("Insira sua API Key na sidebar.")
        else:
            cfg = snapshot()
            profile, tin = st.session_state.bench_profile, st.session_state.bench_in
            tout, rounds = st.session_state.bench_out, st.session_state.bench_rounds
            nocache = st.session_state.bench_nocache
            rows, raw = [], []
            bar = st.progress(0.0, text="Iniciando…")
            done_before = 0
            for lv in levels:
                n = lv * rounds
                prompts = [build_prompt(profile, tin, nocache) for _ in range(n)]

                def on_progress(i, total, lv=lv, base=done_before):
                    bar.progress((base + i) / total_reqs,
                                 text=f"Concorrência {lv}: {i}/{total} requisições")

                records, wall = run_level(cfg, lv, prompts, tout, on_progress)
                rows.append(summarize(lv, records, wall))
                raw.append({"level": lv, "wall": wall, "records": records})
                done_before += n
            bar.empty()
            st.session_state.bench_rows = rows
            st.session_state.bench_raw = raw
            st.session_state.bench_info = {
                "endpoint": cfg["base"], "model": cfg["model"], "profile": profile,
                "tokens_in_target": tin, "max_tokens": tout, "rounds": rounds,
                "prefix_cache_bypass": nocache,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

    rows = st.session_state.bench_rows
    if rows:
        df = pd.DataFrame(rows)
        st.divider()
        peak = df.loc[df["Vazão agregada (tok/s)"].idxmax()]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pico de vazão", f"{peak['Vazão agregada (tok/s)']:.0f} tok/s",
                  help=f"Atingido com concorrência {int(peak['Concorrência'])}")
        m2.metric("Concorrência no pico", int(peak["Concorrência"]))
        first = df.iloc[0]
        m3.metric(f"TTFT p95 (conc. {int(first['Concorrência'])})",
                  f"{first['TTFT p95 (ms)']:.0f} ms" if pd.notna(first["TTFT p95 (ms)"]) else "—")
        m4.metric("Erros / Retries", f"{int(df['Erros'].sum())} / {int(df['Retries'].sum())}")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=df["Concorrência"], y=df["Vazão agregada (tok/s)"], mode="lines+markers",
            name="Vazão agregada (tok/s)", line=dict(color="#7c6ff7", width=3)),
            secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df["Concorrência"], y=df["TTFT p95 (ms)"], mode="lines+markers",
            name="TTFT p95 (ms)", line=dict(color="#f59e0b", width=3, dash="dot")),
            secondary_y=True)
        fig.update_layout(title="Vazão × latência por nível de concorrência",
                          xaxis_title="Concorrência (requisições simultâneas)",
                          legend=dict(orientation="h", y=-0.25), height=380,
                          margin=dict(t=50, b=10))
        fig.update_xaxes(type="category")
        fig.update_yaxes(title_text="tok/s", secondary_y=False)
        fig.update_yaxes(title_text="ms", secondary_y=True)
        st.plotly_chart(fig, width="stretch")

        raw = st.session_state.bench_raw
        g1, g2 = st.columns(2)
        with g1:
            f2 = go.Figure()
            for item in raw:
                ys = [r["ttft"] * 1000 for r in item["records"]
                      if r["ok"] and r.get("ttft") is not None]
                f2.add_trace(go.Box(y=ys, name=str(item["level"]),
                                    marker_color="#7c6ff7", boxmean=True))
            f2.update_xaxes(type="category")
            f2.update_layout(title="Distribuição de TTFT (ms)", showlegend=False,
                             xaxis_title="Concorrência", height=320, margin=dict(t=50, b=10))
            st.plotly_chart(f2, width="stretch")
        with g2:
            f3 = go.Figure()
            for item in raw:
                ys = [r["tps"] for r in item["records"] if r["ok"] and r.get("tps")]
                f3.add_trace(go.Box(y=ys, name=str(item["level"]),
                                    marker_color="#2dd4bf", boxmean=True))
            f3.update_xaxes(type="category")
            f3.update_layout(title="Velocidade por requisição (tok/s)", showlegend=False,
                             xaxis_title="Concorrência", height=320, margin=dict(t=50, b=10))
            st.plotly_chart(f3, width="stretch")

        st.dataframe(df, width="stretch", hide_index=True)
        info = st.session_state.bench_info
        st.caption(f"Perfil **{info.get('profile')}** · modelo `{info.get('model')}` · "
                   f"{info.get('timestamp')}")
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("⬇ CSV", df.to_csv(index=False), "benchmark.csv", "text/csv")
        d2.download_button(
            "⬇ JSON",
            json.dumps({"info": info, "summary": rows, "raw": raw},
                       ensure_ascii=False, indent=2, default=str),
            "benchmark.json", "application/json")
    else:
        st.info("Nenhum resultado ainda. Ajuste o perfil e clique em **Executar benchmark**.")

# ── Aba Custo ─────────────────────────────────────────────────────────────────
with tab_cost:
    st.caption(
        "Compara o custo por token em uma cloud pública com o custo de rodar a mesma "
        "carga on-prem. **Os preços e o custo da GPU são premissas editáveis, "
        "não medições** — ajuste ao cenário do cliente."
    )
    st.markdown("##### ☁️ Cloud pública (US$ por 1M de tokens)")
    p1, p2, p3 = st.columns(3)
    p1.selectbox("Referência de preço", list(CLOUD_PRESETS), key="price_preset",
                 on_change=apply_cloud_preset,
                 help="Valores ilustrativos — não são a tabela de preços de nenhum provedor.")
    p2.number_input("Entrada (US$ / 1M tokens)", key="price_in_per_1m",
                    min_value=0.0, step=0.1, format="%.2f")
    p3.number_input("Saída (US$ / 1M tokens)", key="price_out_per_1m",
                    min_value=0.0, step=0.1, format="%.2f")

    k = session_kpis()
    if k["n"]:
        avg = k["cost"] / k["n"]
        q1, q2, q3 = st.columns(3)
        q1.metric("Custo médio por troca de mensagem", fmt_cost(avg))
        q2.metric("Custo desta sessão", fmt_cost(k["cost"]))
        q3.metric("Projeção: 1.000 trocas", f"US$ {avg * 1000:.2f}")
    else:
        st.info("Converse na aba **Chat** para ver o custo por troca de mensagem.")

    st.markdown("##### 🖥️ On-prem (GPU própria) × Cloud")
    o1, o2, o3 = st.columns(3)
    o1.number_input("Custo por GPU-hora (US$)", key="gpu_hour",
                    min_value=0.0, step=0.5, format="%.2f",
                    help="Amortização + energia + operação. Premissa sua.")
    o2.number_input("Nº de GPUs do endpoint", key="gpu_count", min_value=1, step=1)
    o3.number_input("Volume mensal de tokens de saída (milhões)",
                    key="monthly_out_tokens_m", min_value=0.0, step=50.0)

    rows = st.session_state.bench_rows
    peak_tps = max((r["Vazão agregada (tok/s)"] or 0 for r in rows), default=0)
    if peak_tps <= 0:
        st.info("Rode o **Benchmark** para usar a vazão medida (pico de tok/s) "
                "no cálculo on-prem.")
    else:
        fixed_month = st.session_state.gpu_hour * st.session_state.gpu_count * 730
        capacity_m = peak_tps * 3600 * 730 / 1e6
        onprem_per_m = (st.session_state.gpu_hour * st.session_state.gpu_count
                        / (peak_tps * 3600) * 1e6)
        vol = st.session_state.monthly_out_tokens_m
        cloud_month = vol * st.session_state.price_out_per_1m
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Vazão de pico medida", f"{peak_tps:.0f} tok/s")
        r2.metric("On-prem: US$ / 1M tokens de saída", f"{onprem_per_m:.2f}",
                  delta=f"{onprem_per_m - st.session_state.price_out_per_1m:+.2f} vs cloud",
                  delta_color="inverse")
        r3.metric("On-prem: custo fixo mensal", f"US$ {fixed_month:,.0f}")
        r4.metric(f"Cloud: {vol:,.0f}M tokens/mês", f"US$ {cloud_month:,.0f}")
        be = (fixed_month / st.session_state.price_out_per_1m
              if st.session_state.price_out_per_1m else None)
        msg = (f"Capacidade on-prem estimada: **{capacity_m:,.0f}M tokens de saída/mês** "
               f"(pico medido × 730 h).")
        if be:
            msg += f" Break-even contra a cloud: **{be:,.0f}M tokens/mês**."
        st.caption(msg)
        st.caption(
            "Simplificações: considera apenas tokens de saída, GPU 100% utilizada na "
            "vazão de pico medida e ignora tokens de entrada, ociosidade e custo de rede."
        )

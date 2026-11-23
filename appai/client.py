"""Cliente HTTP para endpoints OpenAI-compatible (vLLM / Nutanix Enterprise AI).

Nada aqui usa st.session_state: as funções recebem um `cfg` (dict) e podem
rodar em threads de benchmark.

cfg = {base, headers, verify, model, temperature, max_retries, prices, extra}
"""

import json
import random
import time

import requests

from appai.cost import estimate_cost

AUTH_TYPES = ["Authorization: Bearer {key}", "x-api-key: {key}", "None / No Auth"]


def api_base(url: str) -> str:
    base = (url or "").strip().rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def build_headers(key: str, auth_type: str) -> dict:
    h = {"Content-Type": "application/json"}
    key = (key or "").strip()
    if auth_type.startswith("Authorization"):
        h["Authorization"] = f"Bearer {key}"
    elif auth_type.startswith("x-api-key"):
        h["x-api-key"] = key
    return h


def needs_key(auth_type: str) -> bool:
    return not auth_type.startswith("None")


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
    merged = [dict(filtered[0])]
    for msg in filtered[1:]:
        if msg["role"] == merged[-1]["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(dict(msg))
    while merged and merged[0]["role"] != "user":
        merged.pop(0)
    return merged


def fetch_models(cfg: dict) -> list:
    r = requests.get(cfg["base"] + "/models", headers=cfg["headers"],
                     timeout=10, verify=cfg["verify"])
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


def is_hibernated(status: int, body: str) -> bool:
    return status == 404 and "hibernat" in (body or "").lower()


def post_with_retry(cfg: dict, body: dict, stream: bool, notify=None, log=None,
                    path: str = "/chat/completions"):
    """POST com retry exponencial + jitter. Retorna (Response, nº de retries).
    notify(msg) é chamado a cada nova tentativa; log (lista) recebe o detalhe
    de cada falha."""
    endpoint = cfg["base"] + path
    max_retries = max(1, int(cfg.get("max_retries", 1)))
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
                hib = "endpoint hibernado no pool" if is_hibernated(resp.status_code, err_body) \
                      else f"HTTP {resp.status_code}"
                notify(f"⏳ {hib}. Nova tentativa {attempt + 1}/{max_retries} "
                       f"em {wait:.1f}s…")
            resp.close()
            time.sleep(wait)
            continue
        resp.raise_for_status()
    resp.raise_for_status()


def chat_body(cfg: dict, messages: list, max_tokens: int, stream: bool = True) -> dict:
    body = {
        "model": cfg["model"],
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": cfg.get("temperature", 0.7),
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    body.update(cfg.get("extra") or {})
    return body


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
    meta["in"] = usage.get("prompt_tokens", 0) or 0
    meta["out"] = out_tokens
    meta["usage_reported"] = bool(usage)
    ttft = metrics.get("ttft")
    end, start = metrics.get("end"), metrics.get("start")
    if ttft is not None:
        meta["ttft"] = ttft
    if end and start and ttft is not None and out_tokens > 1:
        gen_time = max(end - start - ttft, 1e-6)
        meta["tps"] = out_tokens / gen_time
    t_first, t_last = metrics.get("t_first"), metrics.get("t_last")
    chunks = metrics.get("chunks", 0)
    if t_first and t_last and chunks > 1:
        # ITL por chunk SSE; quando o servidor agrupa tokens por chunk, usa a
        # contagem real de tokens para não superestimar a latência entre tokens.
        steps = max(chunks, out_tokens) - 1
        meta["itl"] = (t_last - t_first) / steps * 1000  # ms entre tokens
    meta["cost"] = estimate_cost(meta["in"], meta["out"], prices)
    return meta


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


def stream_collect(cfg: dict, messages: list, max_tokens: int) -> tuple:
    """Faz uma chamada em streaming e devolve (texto, meta). Usado pelo modo
    lado a lado e pelo RAG, onde o texto é exibido só no final."""
    body = chat_body(cfg, messages, max_tokens, stream=True)
    metrics = {"start": time.perf_counter()}
    resp, retries = post_with_retry(cfg, body, stream=True)
    text = "".join(stream_response(resp, metrics))
    meta = finalize_metrics(metrics, cfg["model"], cfg["prices"])
    meta["retries"] = retries
    meta["latency"] = metrics["end"] - metrics["start"]
    return text, meta


def embed(cfg: dict, inputs: list, model: str) -> list:
    """POST /v1/embeddings. Devolve a lista de vetores na ordem de `inputs`."""
    resp, _ = post_with_retry(cfg, {"model": model, "input": inputs},
                              stream=False, path="/embeddings")
    data = resp.json().get("data", [])
    data = sorted(data, key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]

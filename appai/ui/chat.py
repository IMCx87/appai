import json
import statistics
import time

import requests
import streamlit as st

from appai.client import (call_blocking, chat_body, finalize_metrics, post_with_retry,
                          redact, sanitize_messages, stream_response)
from appai.cost import fmt_cost
from appai.ui.state import missing_key, snapshot


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


def render_kpis(slot):
    """Faixa de KPIs da sessão. Chamada de novo após cada resposta do chat para
    não ficar uma rodada defasada."""
    k = session_kpis()
    with slot.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Respostas", k["n"])
        c2.metric("TTFT médio", f"{k['ttft']:.0f} ms" if k["ttft"] is not None else "—")
        c3.metric("Tokens/s médio", f"{k['tps']:.1f}" if k["tps"] is not None else "—")
        c4.metric("Tokens in / out", f"{k['tin']} / {k['tout']}")
        c5.metric("Custo cloud acumulado", fmt_cost(k["cost"]) if k["n"] else "—")


def build_messages() -> list:
    s = st.session_state
    msgs = []
    if s.system_prompt.strip():
        msgs.append({"role": "system", "content": s.system_prompt.strip()})
    msgs.extend(sanitize_messages(s.messages))
    return msgs


def render(kpi_slot):
    s = st.session_state
    for msg in s.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(format_meta(msg["meta"]))

    if s.show_debug and s.debug_log:
        with st.expander("🐛 Debug, última requisição", expanded=False):
            st.code(s.debug_log, language="http")

    prompt = st.chat_input("Digite sua mensagem…")
    if not prompt:
        return
    if missing_key():
        st.error("Insira sua API Key na sidebar.")
        return

    s.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    cfg = snapshot()
    body = chat_body(cfg, build_messages(), s.max_tokens, stream=s.use_stream)
    s.debug_log = (f"POST {cfg['base']}/chat/completions\n"
                   f"Headers: {json.dumps(redact(cfg['headers']), indent=2)}\n"
                   f"Body: {json.dumps(body, indent=2, ensure_ascii=False)}\n")
    req_log: list = []

    with st.chat_message("assistant", avatar="🤖"):
        notice = st.empty()
        try:
            if s.use_stream:
                metrics: dict = {"start": time.perf_counter()}
                resp, _ = post_with_retry(cfg, body, stream=True, notify=notice.info,
                                          log=req_log)
                notice.empty()
                content = st.write_stream(stream_response(resp, metrics))
                meta = finalize_metrics(metrics, cfg["model"], cfg["prices"])
                req_log.append(f"Status: 200 (stream)\nUsage: {json.dumps(metrics.get('usage'))}")
            else:
                with st.spinner("Aguardando resposta…"):
                    content, meta = call_blocking(cfg, body, notify=notice.info, log=req_log)
                notice.empty()
                st.markdown(content)
            st.caption(format_meta(meta))
            s.messages.append({"role": "assistant", "content": content, "meta": meta})
        except requests.exceptions.HTTPError as e:
            notice.empty()
            st.error(f"Erro HTTP {e.response.status_code} após {s.max_retries} "
                     f"tentativa(s): {e.response.text[:500]}")
        except requests.exceptions.ConnectionError as e:
            notice.empty()
            st.error(f"Endpoint inacessível: {e}")
        except Exception as e:
            notice.empty()
            st.error(f"Erro: {e}")
        finally:
            if req_log:
                s.debug_log += "\n" + "\n".join(req_log) + "\n"
    render_kpis(kpi_slot)

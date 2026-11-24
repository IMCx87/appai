import time

import pandas as pd
import streamlit as st

from appai import charts
from appai.client import api_base, build_headers
from appai.rag import (SAMPLE_CORPUS, SAMPLE_QUESTIONS, RagIndex, run_query, run_rag_bench,
                       summarize_rag)
from appai.ui.state import get_store, missing_key, snapshot
from appai.version import build_info

CORPUS_SOURCES = ["Exemplo embutido (infra de IA)", "Meus documentos (.txt / .md)"]


def emb_cfg() -> dict:
    s = st.session_state
    cfg = snapshot()
    if s.rag_emb_url.strip():
        cfg["base"] = api_base(s.rag_emb_url)
    if s.rag_emb_key.strip():
        cfg["headers"] = build_headers(s.rag_emb_key, s.auth_type)
    return cfg


def questions() -> list:
    qs = [q.strip() for q in st.session_state.rag_questions.splitlines() if q.strip()]
    return qs or SAMPLE_QUESTIONS


def render():
    s = st.session_state
    st.caption(
        "Pipeline RAG medido por etapa: **embedding da pergunta → busca vetorial → "
        "prefill → geração**. O TTFT que o usuário sente é a soma das três primeiras, e o "
        "contexto recuperado deixa o prompt bem maior que um chat."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Modelo de embedding", key="rag_emb_model",
                      placeholder="ex.: nai-embed / bge-large / e5-mistral")
        st.text_input("URL de embeddings (vazio = mesmo endpoint)", key="rag_emb_url")
        st.text_input("API Key de embeddings (vazio = mesma)", key="rag_emb_key",
                      type="password")
    with c2:
        st.radio("Corpus", CORPUS_SOURCES, key="rag_corpus_src")
        files = None
        if s.rag_corpus_src.startswith("Meus"):
            files = st.file_uploader("Documentos", type=["txt", "md"],
                                     accept_multiple_files=True)
        st.slider("Palavras por chunk", 40, 400, key="rag_chunk", step=20)
        build = st.button("📚 Indexar corpus", width="stretch",
                          disabled=not s.rag_emb_model.strip())
    with c3:
        st.slider("Chunks recuperados (top-k)", 1, 10, key="rag_k")
        st.slider("Perguntas no teste", 1, 200, key="rag_n")
        st.slider("Concorrência", 1, 32, key="rag_conc")
        st.slider("max_tokens da resposta", 32, 1024, key="rag_max_tokens", step=32)

    if build:
        docs = SAMPLE_CORPUS
        if files:
            docs = [(f.name, f.getvalue().decode("utf-8", errors="replace")) for f in files]
        try:
            idx = RagIndex()
            with st.spinner("Gerando embeddings do corpus…"):
                idx.build(emb_cfg(), s.rag_emb_model.strip(), docs, int(s.rag_chunk))
            s.rag_index = idx
        except Exception as e:
            st.error(f"Falha ao indexar: {e}")

    idx = s.rag_index
    if not idx:
        st.info("Informe o modelo de embedding e clique em **Indexar corpus**. "
                "A busca vetorial roda em memória (numpy) — mede o pipeline e o endpoint "
                "de embeddings, não um banco vetorial de produção.")
        return
    st.success(f"Índice: {idx.stats['chunks']} chunks de {idx.stats['docs']} documento(s), "
               f"dimensão {idx.stats['dim']}, indexado em {idx.stats['seconds']:.1f} s "
               f"({idx.stats['chunks_per_s'] or 0:.0f} chunks/s de embedding).")

    st.text_area("Perguntas (uma por linha; vazio = perguntas de exemplo)", key="rag_questions",
                 height=90, placeholder="\n".join(SAMPLE_QUESTIONS[:3]))
    q1, q2, q3 = st.columns([2, 1, 1])
    q1.text_input("Rótulo da execução", key="rag_label")
    try_one = q2.button("💬 Testar 1 pergunta", width="stretch")
    run = q3.button("▶ Rodar teste RAG", type="primary", width="stretch")

    if (try_one or run) and missing_key():
        st.error("Insira sua API Key na sidebar.")
        return
    if try_one:
        rec = run_query(snapshot(), emb_cfg(), s.rag_emb_model.strip(), idx, questions()[0],
                        int(s.rag_k), int(s.rag_max_tokens), keep_text=True)
        if rec["ok"]:
            st.markdown(f"**{rec['question']}**\n\n{rec['answer']}")
            st.caption(f"Fontes: {', '.join(rec['sources'])} · TTFT percebido "
                       f"{rec['user_ttft_ms']:.0f} ms · total {rec['total_ms']:.0f} ms · "
                       f"{rec['tokens_in']} tokens de entrada")
        else:
            st.error(rec.get("error"))
    if run:
        bar = st.progress(0.0)
        recs = run_rag_bench(snapshot(), emb_cfg(), s.rag_emb_model.strip(), idx, questions(),
                             int(s.rag_n), int(s.rag_conc), int(s.rag_k),
                             int(s.rag_max_tokens),
                             lambda i, n: bar.progress(i / n, text=f"{i}/{n} perguntas"))
        bar.empty()
        summ = summarize_rag(recs)
        b = build_info()
        info = {"endpoint": snapshot()["base"], "model": s.model_name,
                "emb_model": s.rag_emb_model, "k": s.rag_k, "concurrency": s.rag_conc,
                "chunks": idx.stats["chunks"], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "app": f"v{b['version']} ({b['commit']})"}
        label = s.rag_label.strip() or f"{s.model_name} + {s.rag_emb_model} · k={s.rag_k}"
        data = {"summary": summ, "records": recs, "index": idx.stats}
        get_store().save("rag", label, info, data)
        s.rag_result = {"label": label, "info": info, "data": data}

    r = s.rag_result
    if r:
        summ = r["data"]["summary"]
        st.divider()
        by = {x["key"]: x for x in summ["stages"]}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("TTFT percebido p50", f"{by['user_ttft_ms']['p50 (ms)'] or 0:.0f} ms")
        m2.metric("TTFT percebido p95", f"{by['user_ttft_ms']['p95 (ms)'] or 0:.0f} ms")
        m3.metric("Tokens de entrada (média)", f"{summ['tokens_in_mean'] or 0:.0f}")
        m4.metric("Erros", f"{summ['errors']} / {summ['requests']}")
        st.plotly_chart(charts.rag_stages(summ), width="stretch", key="rag-stages")
        st.dataframe(pd.DataFrame(summ["stages"]).drop(columns=["key"]), width="stretch",
                     hide_index=True)

"""RAG medido por etapa: embedding da pergunta → busca vetorial → geração.

O "TTFT que o usuário sente" num assistente RAG é embedding + busca + TTFT do
modelo. Medir cada etapa separadamente mostra onde está o tempo — e o prompt
com contexto recuperado é bem maior que um chat, o que pesa no prefill.

A busca vetorial aqui é em memória (numpy, força bruta por cosseno): mede o
pipeline e o endpoint de embeddings, não um banco vetorial de produção.
"""

import concurrent.futures as cf
import random
import statistics
import time

import numpy as np

from appai.bench import pct
from appai.client import chat_body, embed, finalize_metrics, post_with_retry, stream_response

SAMPLE_CORPUS = [
    ("KV cache", "Durante a inferência de um LLM, o KV cache guarda as chaves e valores de "
     "atenção de cada token já processado. Ele cresce com o tamanho do contexto e com o "
     "número de requisições simultâneas, e vive na memória da GPU. Quando o KV cache "
     "enche, servidores como o vLLM passam a enfileirar requisições ou a preemptar as "
     "que estão em andamento, o que aumenta a latência de cauda. Quantizar o KV cache "
     "em FP8 ou reduzir o tamanho máximo de contexto libera memória para mais usuários."),
    ("Prefill e decode", "A geração de texto tem duas fases. No prefill o modelo processa "
     "o prompt inteiro de uma vez; é uma fase limitada por computação e determina o tempo "
     "até o primeiro token (TTFT). No decode o modelo gera um token por vez; é uma fase "
     "limitada por largura de banda de memória e determina a latência entre tokens (ITL). "
     "Prompts longos, como os de RAG, aumentam o custo de prefill."),
    ("Continuous batching", "O continuous batching permite que o servidor de inferência "
     "adicione novas requisições ao lote a cada passo de decode, em vez de esperar o lote "
     "inteiro terminar. Isso aumenta muito a vazão agregada, mas cada requisição divide a "
     "GPU com as demais: com mais usuários simultâneos a vazão total sobe até saturar, "
     "enquanto a velocidade individual de cada resposta cai."),
    ("Prefix caching", "O prefix caching reaproveita o KV cache de prompts que começam com "
     "o mesmo texto, como um system prompt longo ou o mesmo documento consultado várias "
     "vezes. O prefill desse trecho deixa de ser recalculado, reduzindo o TTFT e o uso de "
     "GPU. Benchmarks que repetem o mesmo prompt podem parecer melhores do que a carga "
     "real por causa desse efeito."),
    ("Storage para modelos", "Os pesos de um modelo de linguagem ocupam de alguns GB a "
     "centenas de GB. Ao iniciar ou acordar um endpoint, esses pesos são lidos do storage "
     "e carregados na GPU. A vazão de leitura sequencial do storage — NFS, iSCSI, objeto "
     "ou NVMe local — influencia diretamente o tempo de cold start e a agilidade para "
     "escalar réplicas sob demanda."),
    ("Snapshots e proteção de dados", "Snapshots de volume permitem capturar o estado de "
     "um banco de dados ou de um repositório de modelos em um ponto no tempo, com pouco "
     "impacto de desempenho. Combinados com replicação para outro site, formam a base de "
     "uma estratégia de recuperação de desastres com RPO e RTO definidos. Em Kubernetes, "
     "o CSI expõe snapshots como objetos VolumeSnapshot."),
    ("Kubernetes e GPUs", "Em Kubernetes, GPUs são expostas por device plugins e "
     "requisitadas pelos pods como recursos. O NVIDIA GPU Operator instala drivers, o "
     "device plugin e o exportador DCGM, que publica métricas como utilização, memória e "
     "potência para o Prometheus. Recursos como MIG e time-slicing permitem dividir uma "
     "GPU entre várias cargas."),
    ("Soberania de dados", "Rodar modelos on-premises mantém prompts, documentos e respostas "
     "dentro do perímetro da empresa, o que simplifica requisitos de privacidade e "
     "regulação. A comparação de custo com APIs de nuvem depende do volume: com uso alto e "
     "constante, GPUs próprias tendem a ter custo por token menor; com uso baixo e "
     "esporádico, pagar por token pode ser mais barato."),
    ("Rede para inferência", "Para inferência em um único nó, a rede raramente é o "
     "gargalo: o tráfego por requisição é pequeno. Ela passa a importar em modelos "
     "distribuídos entre várias GPUs ou nós, com tensor ou pipeline parallelism, onde "
     "interconexões como NVLink e RDMA reduzem a latência de comunicação entre GPUs."),
    ("SLO de inferência", "Um SLO de inferência costuma combinar o p95 do tempo até o "
     "primeiro token, que define a sensação de resposta imediata, com o p95 da latência "
     "entre tokens, que define a fluidez da leitura. A capacidade útil de um endpoint é o "
     "número de usuários simultâneos que ele atende sem violar esses limites, não a vazão "
     "máxima que ele alcança."),
]

SAMPLE_QUESTIONS = [
    "O que acontece quando o KV cache enche?",
    "Qual a diferença entre prefill e decode?",
    "Por que a vazão agregada sobe e a velocidade individual cai?",
    "Como o prefix caching afeta o TTFT?",
    "Por que o storage influencia o cold start de um modelo?",
    "Como snapshots ajudam na recuperação de desastres?",
    "Quais métricas o DCGM publica?",
    "Quando GPUs próprias são mais baratas que API de nuvem?",
    "Quando a rede vira gargalo na inferência?",
    "Como definir a capacidade útil de um endpoint de inferência?",
]


def chunk_text(text: str, size: int = 120, overlap: int = 20) -> list:
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - overlap, step)]


class RagIndex:
    def __init__(self):
        self.chunks: list = []
        self.matrix: np.ndarray | None = None
        self.stats: dict = {}

    def build(self, cfg: dict, model: str, docs: list, chunk_size: int = 120,
              batch: int = 32) -> dict:
        self.chunks = [{"doc": title, "text": c}
                       for title, text in docs for c in chunk_text(text, chunk_size)]
        if not self.chunks:
            raise ValueError("Corpus vazio")
        t0 = time.perf_counter()
        vecs = []
        for i in range(0, len(self.chunks), batch):
            vecs.extend(embed(cfg, [c["text"] for c in self.chunks[i:i + batch]], model))
        elapsed = time.perf_counter() - t0
        m = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        self.matrix = m / np.where(norms == 0, 1, norms)
        self.stats = {"chunks": len(self.chunks), "dim": int(m.shape[1]),
                      "seconds": elapsed, "chunks_per_s": len(self.chunks) / elapsed
                      if elapsed > 0 else None, "docs": len(docs)}
        return self.stats

    def search(self, qvec, k: int) -> list:
        q = np.asarray(qvec, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1)
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


def build_rag_prompt(question: str, hits: list) -> str:
    ctx = "\n\n".join(f"[{i + 1}] ({h['doc']}) {h['text']}" for i, h in enumerate(hits))
    return ("Responda à pergunta usando apenas o contexto abaixo. Cite as fontes "
            f"pelo número.\n\nContexto:\n{ctx}\n\nPergunta: {question}")


def run_query(cfg_chat: dict, cfg_emb: dict, emb_model: str, index: RagIndex,
              question: str, k: int, max_tokens: int, keep_text: bool = False) -> dict:
    rec = {"question": question, "ok": False}
    t0 = time.perf_counter()
    try:
        qvec = embed(cfg_emb, [question], emb_model)[0]
        t1 = time.perf_counter()
        hits = index.search(qvec, k)
        t2 = time.perf_counter()
        body = chat_body(cfg_chat, [{"role": "user",
                                     "content": build_rag_prompt(question, hits)}],
                         max_tokens, stream=True)
        metrics = {"start": t2}
        resp, retries = post_with_retry(cfg_chat, body, stream=True)
        text = "".join(stream_response(resp, metrics))
        meta = finalize_metrics(metrics, cfg_chat["model"], cfg_chat["prices"])
        t3 = metrics["end"]
        ttft = meta.get("ttft") or 0
        rec.update(
            ok=True,
            embed_ms=(t1 - t0) * 1000,
            retrieve_ms=(t2 - t1) * 1000,
            prefill_ms=ttft * 1000,
            decode_ms=max(t3 - t2 - ttft, 0) * 1000,
            user_ttft_ms=(t2 - t0 + ttft) * 1000,
            total_ms=(t3 - t0) * 1000,
            tokens_in=meta.get("in"), tokens_out=meta.get("out"),
            cost=meta.get("cost", 0.0), retries=retries,
            sources=[h["doc"] for h in hits],
            top_score=hits[0]["score"] if hits else None,
        )
        if keep_text:
            rec["answer"] = text
            rec["hits"] = hits
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)[:200]
        rec["total_ms"] = (time.perf_counter() - t0) * 1000
    return rec


STAGES = [("embed_ms", "Embedding"), ("retrieve_ms", "Busca vetorial"),
          ("prefill_ms", "Prefill (TTFT do modelo)"), ("decode_ms", "Geração (decode)")]


def run_rag_bench(cfg_chat, cfg_emb, emb_model, index, questions, n, concurrency,
                  k, max_tokens, on_progress, seed: int = 7) -> list:
    rng = random.Random(seed)
    qs = [rng.choice(questions) for _ in range(n)]
    out = []
    with cf.ThreadPoolExecutor(max_workers=max(concurrency, 1)) as ex:
        futs = [ex.submit(run_query, cfg_chat, cfg_emb, emb_model, index, q, k, max_tokens)
                for q in qs]
        for i, f in enumerate(cf.as_completed(futs), 1):
            out.append(f.result())
            on_progress(i, n)
    return out


def summarize_rag(records: list) -> dict:
    ok = [r for r in records if r.get("ok")]
    s = {"requests": len(records), "errors": len(records) - len(ok), "stages": []}
    for key, label in STAGES + [("user_ttft_ms", "TTFT percebido (embedding + busca + prefill)"),
                                ("total_ms", "Total")]:
        vals = [r[key] for r in ok if r.get(key) is not None]
        s["stages"].append({
            "key": key, "Etapa": label,
            "p50 (ms)": round(pct(vals, 50), 1) if vals else None,
            "p95 (ms)": round(pct(vals, 95), 1) if vals else None,
            "média (ms)": round(statistics.fmean(vals), 1) if vals else None,
        })
    tin = [r["tokens_in"] for r in ok if r.get("tokens_in")]
    s["tokens_in_mean"] = statistics.fmean(tin) if tin else None
    return s

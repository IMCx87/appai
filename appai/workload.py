"""Geração de carga: prompts sintéticos, distribuição de tamanhos e datasets.

Contagem de tokens é aproximada (~0,75 palavra por token, bom para PT/EN);
o número real vem do `usage` devolvido pelo servidor.
"""

import io
import json
import math
import random
import uuid

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

DISTRIBUTIONS = ["Fixo", "Lognormal (realista)"]
WORDS_PER_TOKEN = 0.75
MAX_TOKENS_CAP = 32768


def approx_tokens(text: str) -> int:
    return max(1, round(len(text.split()) / WORDS_PER_TOKEN))


def build_prompt(question: str, target_tokens: int, nocache: bool,
                 rng: random.Random | None = None) -> str:
    """Monta um prompt com ~target_tokens tokens de contexto + a pergunta.
    Com nocache, um prefixo único por requisição evita prefix caching no servidor."""
    words = FILLER.split()
    n_words = max(0, int(target_tokens * WORDS_PER_TOKEN) - len(question.split()))
    reps, rest = divmod(n_words, len(words))
    context = FILLER * reps + " ".join(words[:rest])
    tag = (rng.getrandbits(32) if rng else uuid.uuid4().int & 0xFFFFFFFF)
    prefix = f"[ref {tag:08x}]\n" if nocache else ""
    return f"{prefix}{context}\n\n{question}".strip()


def lognormal(mean: float, cv: float, rng: random.Random) -> float:
    """Amostra lognormal com média `mean` e coeficiente de variação `cv`."""
    if cv <= 0:
        return mean
    sigma2 = math.log(1 + cv * cv)
    mu = math.log(mean) - sigma2 / 2
    return rng.lognormvariate(mu, math.sqrt(sigma2))


def sample_size(mean: int, dist: str, cv: float, rng: random.Random) -> int:
    if dist.startswith("Lognormal"):
        v = lognormal(max(mean, 1), cv, rng)
        return int(min(max(round(v), 8), max(mean * 6, 16), MAX_TOKENS_CAP))
    return int(mean)


def make_requests(n: int, spec: dict, rng: random.Random) -> list:
    """Gera n requisições {prompt, max_tokens}.

    spec: question, in, out, dist, cv, nocache, dataset (lista opcional de
    {prompt, out}). Com dataset, amostra prompts reais em vez dos sintéticos.
    """
    dataset = spec.get("dataset") or []
    reqs = []
    for _ in range(n):
        if dataset:
            item = rng.choice(dataset)
            prompt = item["prompt"]
            if spec.get("nocache"):
                prompt = f"[ref {rng.getrandbits(32):08x}]\n{prompt}"
            out = item.get("out") or spec["out"]
            reqs.append({"prompt": prompt, "max_tokens": int(min(out, MAX_TOKENS_CAP))})
            continue
        tin = sample_size(spec["in"], spec.get("dist", "Fixo"), spec.get("cv", 0.6), rng)
        tout = sample_size(spec["out"], spec.get("dist", "Fixo"), spec.get("cv", 0.6), rng)
        reqs.append({
            "prompt": build_prompt(spec["question"], tin, spec.get("nocache", True), rng),
            "max_tokens": tout,
        })
    return reqs


def _sharegpt_pairs(items: list) -> list:
    out = []
    for it in items:
        conv = it.get("conversations") or it.get("conversation") or []
        human = next((c for c in conv if c.get("from") in ("human", "user")), None)
        gpt = next((c for c in conv if c.get("from") in ("gpt", "assistant")), None)
        if human and human.get("value"):
            o = approx_tokens(gpt["value"]) if gpt and gpt.get("value") else None
            out.append({"prompt": human["value"], "out": o})
    return out


def load_dataset(data: bytes, filename: str, limit: int = 5000) -> list:
    """Carrega prompts reais. Aceita:
    - JSONL com campo "prompt" (e opcional "max_tokens" / "output_tokens");
    - JSON/JSONL no formato ShareGPT ("conversations": [{from, value}, ...]);
    - TXT com um prompt por linha.
    Devolve [{prompt, out}] (out = tokens de saída esperados ou None)."""
    text = data.decode("utf-8", errors="replace")
    name = filename.lower()
    items: list = []
    if name.endswith(".json"):
        parsed = json.loads(text)
        items = parsed if isinstance(parsed, list) else [parsed]
    elif name.endswith(".jsonl"):
        for line in io.StringIO(text):
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    else:
        return [{"prompt": ln.strip(), "out": None}
                for ln in text.splitlines() if ln.strip()][:limit]

    if items and isinstance(items[0], dict) and ("conversations" in items[0]
                                                 or "conversation" in items[0]):
        return _sharegpt_pairs(items)[:limit]
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        prompt = it.get("prompt") or it.get("input") or it.get("question")
        if prompt:
            o = it.get("max_tokens") or it.get("output_tokens") or it.get("output_len")
            out.append({"prompt": str(prompt), "out": int(o) if o else None})
    return out[:limit]


def dataset_stats(dataset: list) -> dict:
    if not dataset:
        return {}
    ins = sorted(approx_tokens(d["prompt"]) for d in dataset)
    outs = sorted(d["out"] for d in dataset if d.get("out"))
    mid = lambda v: v[len(v) // 2] if v else None  # noqa: E731
    return {"n": len(dataset), "in_median": mid(ins), "in_max": ins[-1],
            "out_median": mid(outs), "out_known": len(outs)}

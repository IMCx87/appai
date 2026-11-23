"""Servidor mock OpenAI-compatible que imita um vLLM/NAI — para desenvolver,
testar e ensaiar demos sem GPU.

    python -m appai.mock_server --port 8000
    # no app: API Endpoint URL = http://localhost:8000/v1, auth = None
    # vLLM metrics URL = http://localhost:8000/metrics
    # Prometheus URL   = http://localhost:8000      (DCGM sintético)

Simula: streaming SSE com usage, continuous batching com limite de batch
(fila quando estoura), KV cache finito com preempções, prefix cache,
hibernação (404 "hibernated endpoint" até acordar), /v1/embeddings
determinístico (bag-of-words com hash) e uma API Prometheus mínima com
métricas DCGM sintéticas. Os números são plausíveis, não reais.
"""

import argparse
import hashlib
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

WORD = re.compile(r"\w+", re.UNICODE)
EMB_DIM = 256
LOREM = ("A inferência de modelos de linguagem depende de GPU, memória e storage. "
         "O KV cache cresce com o contexto e com o número de usuários simultâneos. ").split()


class Sim:
    def __init__(self, decode_tps=45.0, prefill_tps=6000.0, max_batch=16,
                 kv_tokens=40000, wake_s=0.0, hibernated=False, natural_out=120,
                 model="mock-llm", emb_model="mock-embed"):
        self.decode_tps, self.prefill_tps = decode_tps, prefill_tps
        self.max_batch, self.kv_tokens = max_batch, kv_tokens
        self.wake_s, self.natural_out = wake_s, natural_out
        self.model, self.emb_model = model, emb_model
        self.lock = threading.Lock()
        self.slots = threading.Semaphore(max_batch)
        self.running = self.waiting = self.kv_used = 0
        self.preempt = self.prefix_hits = self.prefix_queries = 0
        self.prefixes: set = set()
        self.hibernated, self.wake_at = hibernated, None
        self.history: list = []
        self._stop = threading.Event()
        threading.Thread(target=self._sample, daemon=True).start()

    # métricas "DCGM" derivadas do estado
    def gpu_state(self):
        load = min(self.running / max(self.max_batch * 0.5, 1), 1.0)
        util = 0.0 if self.hibernated else 6 + 90 * load
        kv = min(self.kv_used / self.kv_tokens, 1.0)
        mem = 0.0 if self.hibernated else 18000 + 22000 * kv
        power = 25.0 if self.hibernated else 70 + util * 3.2
        return util, mem, power, kv

    def _sample(self):
        while not self._stop.is_set():
            util, mem, power, _ = self.gpu_state()
            self.history.append((time.time(), util, mem, power))
            self.history = self.history[-20000:]
            self._stop.wait(0.5)

    def hibernate(self):
        with self.lock:
            self.hibernated, self.wake_at = True, None

    def check_awake(self) -> bool:
        with self.lock:
            if not self.hibernated:
                return True
            now = time.time()
            if self.wake_at is None:
                self.wake_at = now + self.wake_s
            if now >= self.wake_at:
                self.hibernated, self.wake_at = False, None
                return True
            return False


def approx_tokens(text: str) -> int:
    return max(1, round(len(text.split()) / 0.75))


def embed_text(text: str) -> list:
    v = [0.0] * EMB_DIM
    for w in WORD.findall(text.lower()):
        if len(w) < 3:
            continue
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % EMB_DIM] += 1.0
    return v


def make_handler(sim: Sim):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):  # silencioso
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/v1/models", "/models"):
                return self._json(200, {"object": "list", "data": [
                    {"id": sim.model, "object": "model"},
                    {"id": sim.emb_model, "object": "model"}]})
            if u.path == "/metrics":
                return self._metrics()
            if u.path == "/api/v1/query_range":
                return self._query_range(parse_qs(u.query))
            if u.path in ("/health", "/"):
                return self._json(200, {"status": "ok", "hibernated": sim.hibernated})
            self._json(404, {"message": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            if u.path == "/admin/hibernate":
                sim.hibernate()
                return self._json(200, {"hibernated": True})
            if u.path in ("/v1/embeddings", "/embeddings"):
                body = self._body()
                inputs = body.get("input")
                inputs = [inputs] if isinstance(inputs, str) else inputs or []
                time.sleep(0.004 + 0.0008 * len(inputs))
                return self._json(200, {"object": "list", "model": body.get("model"),
                                        "data": [{"object": "embedding", "index": i,
                                                  "embedding": embed_text(t)}
                                                 for i, t in enumerate(inputs)],
                                        "usage": {"prompt_tokens": sum(
                                            approx_tokens(t) for t in inputs)}})
            if u.path in ("/v1/chat/completions", "/chat/completions"):
                return self._chat(self._body())
            self._json(404, {"message": "not found"})

        def _chat(self, body):
            if not sim.check_awake():
                return self._json(404, {"message": "Request routed to hibernated endpoint"})
            prompt = "\n".join(str(m.get("content", "")) for m in body.get("messages", []))
            tin = approx_tokens(prompt)
            max_tokens = int(body.get("max_tokens") or 256)
            tout = max_tokens if body.get("ignore_eos") else min(max_tokens, sim.natural_out)
            stream = bool(body.get("stream"))
            with sim.lock:
                sim.waiting += 1
            sim.slots.acquire()
            try:
                with sim.lock:
                    sim.waiting -= 1
                    sim.running += 1
                    sim.kv_used += tin + tout
                    pressure = sim.kv_used > sim.kv_tokens
                    if pressure:
                        sim.preempt += 1
                    key = prompt[:64]
                    sim.prefix_queries += tin
                    cached = key in sim.prefixes
                    if cached:
                        sim.prefix_hits += int(tin * 0.8)
                    sim.prefixes.add(key)
                    running = sim.running
                eff_in = tin * (0.2 if cached else 1.0)
                prefill = 0.03 + eff_in / sim.prefill_tps * (1 + 0.15 * (running - 1))
                if pressure:
                    prefill += 0.4
                per_tok = 1.0 / sim.decode_tps * (1 + 0.07 * (running - 1))
                rid = f"chatcmpl-{int(time.time() * 1000)}"
                if not stream:
                    time.sleep(prefill + per_tok * tout)
                    return self._json(200, {
                        "id": rid, "object": "chat.completion", "model": sim.model,
                        "choices": [{"index": 0, "finish_reason": "stop", "message": {
                            "role": "assistant",
                            "content": " ".join(LOREM[i % len(LOREM)] for i in range(tout))}}],
                        "usage": {"prompt_tokens": tin, "completion_tokens": tout,
                                  "total_tokens": tin + tout}})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                time.sleep(prefill)
                for i in range(tout):
                    chunk = {"id": rid, "object": "chat.completion.chunk", "model": sim.model,
                             "choices": [{"index": 0, "delta": {
                                 "content": LOREM[i % len(LOREM)] + " "}}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(per_tok)
                if (body.get("stream_options") or {}).get("include_usage"):
                    usage = {"id": rid, "object": "chat.completion.chunk", "model": sim.model,
                             "choices": [], "usage": {"prompt_tokens": tin,
                                                      "completion_tokens": tout,
                                                      "total_tokens": tin + tout}}
                    self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with sim.lock:
                    sim.running -= 1
                    sim.kv_used -= tin + tout
                sim.slots.release()

        def _metrics(self):
            _, _, _, kv = sim.gpu_state()
            m = f'model_name="{sim.model}"'
            text = (
                "# HELP vllm:num_requests_running Requests running.\n"
                "# TYPE vllm:num_requests_running gauge\n"
                f"vllm:num_requests_running{{{m}}} {sim.running}\n"
                "# TYPE vllm:num_requests_waiting gauge\n"
                f"vllm:num_requests_waiting{{{m}}} {sim.waiting}\n"
                "# TYPE vllm:kv_cache_usage_perc gauge\n"
                f"vllm:kv_cache_usage_perc{{{m}}} {kv:.4f}\n"
                "# TYPE vllm:num_preemptions_total counter\n"
                f"vllm:num_preemptions_total{{{m}}} {sim.preempt}\n"
                "# TYPE vllm:prefix_cache_queries_total counter\n"
                f"vllm:prefix_cache_queries_total{{{m}}} {sim.prefix_queries}\n"
                "# TYPE vllm:prefix_cache_hits_total counter\n"
                f"vllm:prefix_cache_hits_total{{{m}}} {sim.prefix_hits}\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(text)))
            self.end_headers()
            self.wfile.write(text)

        def _query_range(self, qs):
            q = (qs.get("query") or [""])[0].upper()
            start = float((qs.get("start") or ["0"])[0])
            end = float((qs.get("end") or [str(time.time())])[0])
            step = float((qs.get("step") or ["1s"])[0].rstrip("s") or 1)
            col = 1 if "UTIL" in q else 2 if "FB_USED" in q else 3 if "POWER" in q else None
            if col is None:
                return self._json(200, {"status": "success",
                                        "data": {"resultType": "matrix", "result": []}})
            div = 1024.0 if col == 2 and "1024" in q else 1.0
            values, next_t = [], start
            for row in sim.history:
                if start <= row[0] <= end and row[0] >= next_t:
                    values.append([row[0], f"{row[col] / div:.3f}"])
                    next_t = row[0] + step
            return self._json(200, {"status": "success", "data": {
                "resultType": "matrix", "result": [{"metric": {"gpu": "0"}, "values": values}]}})

    return H


def serve(port: int = 8000, host: str = "127.0.0.1", **kw):
    """Sobe o mock em uma thread; devolve (server, sim)."""
    sim = Sim(**kw)
    srv = ThreadingHTTPServer((host, port), make_handler(sim))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, sim


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--decode-tps", type=float, default=45.0, help="tok/s de uma requisição sozinha")
    ap.add_argument("--max-batch", type=int, default=16, help="requisições simultâneas antes de enfileirar")
    ap.add_argument("--kv-tokens", type=int, default=40000, help="capacidade do KV cache em tokens")
    ap.add_argument("--wake-seconds", type=float, default=20.0, help="tempo para acordar da hibernação")
    ap.add_argument("--hibernated", action="store_true", help="inicia hibernado")
    a = ap.parse_args()
    srv, _ = serve(a.port, a.host, decode_tps=a.decode_tps, max_batch=a.max_batch,
                   kv_tokens=a.kv_tokens, wake_s=a.wake_seconds, hibernated=a.hibernated)
    print(f"mock vLLM/NAI em http://{a.host}:{a.port}/v1  (metrics: /metrics, "
          f"prometheus: /api/v1/query_range, hibernar: POST /admin/hibernate)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()

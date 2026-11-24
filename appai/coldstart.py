"""Cold start: quanto tempo um endpoint hibernado (ou escalado a zero) leva
até responder o primeiro token.

Boa parte desse tempo é carregar dezenas de GB de pesos do storage para a
GPU — por isso a medição é rotulada pelo backend de storage (Nutanix Files,
Volumes, disco local...). O tempo também inclui agendamento do pod, pull de
imagem e inicialização do runtime: a vazão de carga derivada é uma estimativa
por baixo, não a vazão do storage.
"""

import time

import requests

from appai.client import chat_body, finalize_metrics, is_retryable, stream_response

STORAGE_BACKENDS = ["Nutanix Files (NFS)", "Nutanix Volumes (iSCSI)",
                    "Nutanix Objects (S3)", "Disco local / NVMe", "Outro"]


def measure_cold_start(cfg: dict, prompt: str = "Responda apenas: OK.",
                       timeout_s: float = 900, poll_s: float = 5,
                       on_event=None, should_stop=None) -> dict:
    """Tenta até o endpoint responder. Registra cada tentativa.

    Retorna {ok, ready_s, ttft_s, total_s, attempts, events, meta, error}.
    ready_s = do primeiro envio até a resposta HTTP de sucesso;
    total_s = do primeiro envio até o primeiro token."""
    t0 = time.perf_counter()
    wall0 = time.time()
    events = []

    def ev(status, note):
        e = {"t": round(time.perf_counter() - t0, 2), "status": status, "note": note}
        events.append(e)
        if on_event:
            on_event(e)

    body = chat_body(cfg, [{"role": "user", "content": prompt}], 16, stream=True)
    attempts = 0
    while True:
        if should_stop and should_stop():
            return _result(False, t0, wall0, attempts, events, error="interrompido")
        elapsed = time.perf_counter() - t0
        if elapsed > timeout_s:
            return _result(False, t0, wall0, attempts, events,
                           error=f"timeout após {timeout_s:.0f}s")
        attempts += 1
        t_try = time.perf_counter()
        try:
            resp = requests.post(cfg["base"] + "/chat/completions", headers=cfg["headers"],
                                 json=body, stream=True, timeout=(10, 300),
                                 verify=cfg["verify"])
        except (requests.ConnectionError, requests.Timeout) as e:
            ev("conn", f"sem conexão ({type(e).__name__})")
            time.sleep(poll_s)
            continue
        if resp.status_code >= 400:
            txt = resp.text[:300]
            resp.close()
            if is_retryable(resp.status_code, txt):
                note = "hibernado/acordando" if "hibernat" in txt.lower() else txt[:80]
                ev(resp.status_code, note)
                time.sleep(poll_s)
                continue
            ev(resp.status_code, txt[:120])
            return _result(False, t0, wall0, attempts, events,
                           error=f"HTTP {resp.status_code}: {txt[:200]}")
        ready = time.perf_counter() - t0
        ev(resp.status_code, "respondeu")
        metrics = {"start": t_try}
        for _ in stream_response(resp, metrics):
            pass
        meta = finalize_metrics(metrics, cfg["model"], cfg["prices"])
        total = (metrics["t_first"] - t0) if metrics.get("t_first") else None
        return _result(True, t0, wall0, attempts, events, ready=ready, total=total,
                       meta=meta)


def _result(ok, t0, wall0, attempts, events, ready=None, total=None, meta=None, error=""):
    return {
        "ok": ok,
        "ready_s": ready,
        "total_s": total,
        "ttft_s": (meta or {}).get("ttft"),
        "attempts": attempts,
        "events": events,
        "meta": meta or {},
        "error": error,
        "started": wall0,
        "elapsed_s": time.perf_counter() - t0,
    }


def load_throughput_gbs(model_gb: float, seconds: float | None) -> float | None:
    """GB/s efetivos (pesos ÷ tempo até pronto). Limite inferior: o tempo
    inclui tudo que não é leitura de storage."""
    if not model_gb or not seconds:
        return None
    return model_gb / seconds

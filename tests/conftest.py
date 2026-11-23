import socket

import pytest

from appai.mock_server import serve


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def mock():
    """Sobe um mock vLLM rápido e devolve (cfg, sim, base_url)."""
    port = free_port()
    srv, sim = serve(port, decode_tps=400, prefill_tps=50000, max_batch=4,
                     kv_tokens=4000, wake_s=0.6)
    root = f"http://127.0.0.1:{port}"
    cfg = {"base": root + "/v1", "headers": {"Content-Type": "application/json"},
           "verify": False, "model": "mock-llm", "temperature": 0.0, "max_retries": 2,
           "prices": (1.0, 2.0), "extra": {}}
    yield cfg, sim, root
    srv.shutdown()

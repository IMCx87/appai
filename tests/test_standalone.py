"""O arquivo único (app_standalone.py) precisa estar em dia com o pacote e
funcionar sozinho, sem o diretório appai/ por perto."""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_standalone_is_up_to_date():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "build_standalone.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_standalone_runs_alone(tmp_path):
    """Copia só o arquivo para uma pasta vazia e roda o app inteiro num processo
    separado (no processo do pytest o pacote real já está importado)."""
    shutil.copy(ROOT / "app_standalone.py", tmp_path)
    script = textwrap.dedent("""
        import socket, sys
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("app_standalone.py", default_timeout=60).run()
        assert not at.exception, at.exception
        import appai
        assert appai.__bundle__["digest"], "pacote não veio do bundle"
        assert "appai.ui.bench" in sys.modules

        from appai.mock_server import serve
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        srv, _ = serve(port, decode_tps=400, prefill_tps=50000, max_batch=4)
        at.session_state["api_url"] = f"http://127.0.0.1:{port}/v1"
        at.session_state["auth_type"] = "None / No Auth"
        at.session_state["model_name"] = "mock-llm"
        at.session_state["bench_levels"] = [1, 2]
        at.session_state["bench_rounds"] = 1
        at.session_state["bench_out"] = 16
        at.run()
        at.button(key="run_bench").click().run()
        assert not at.exception, at.exception
        res = at.session_state["bench_result"]
        assert [lv["load"] for lv in res["data"]["levels"]] == [1, 2] and res["id"]
        sidebar = " ".join(m.value for m in at.sidebar.caption)
        assert "arquivo único" in sidebar, sidebar
        srv.shutdown()
        print("OK")
    """)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["APPAI_DATA_DIR"] = str(tmp_path / "data")
    r = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]
    assert not (tmp_path / "appai").exists()

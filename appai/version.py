"""Versão e commit do build, exibidos na sidebar e gravados em cada execução.

No container os valores vêm de build-args (APP_VERSION, GIT_SHA, BUILD_DATE).
Rodando local a partir do clone, cai para `git rev-parse`.
"""

import os
import subprocess
from functools import lru_cache
from pathlib import Path

VERSION = "2.0.0"


@lru_cache(maxsize=1)
def build_info() -> dict:
    sha = os.environ.get("GIT_SHA", "").strip()
    date = os.environ.get("BUILD_DATE", "").strip()
    source = "container" if sha else "local"
    if not sha:
        try:
            sha = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parent.parent,
                capture_output=True, text=True, timeout=2, check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            sha = ""
    return {
        "version": os.environ.get("APP_VERSION", "").strip() or VERSION,
        "commit": sha[:7] if sha else "desconhecido",
        "build_date": date,
        "source": source,
    }

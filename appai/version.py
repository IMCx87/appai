"""Versão e commit do build, exibidos na sidebar e gravados em cada execução.

Três formas de rodar, três origens:
- container: build-args do CI (APP_VERSION, GIT_SHA, BUILD_DATE);
- arquivo único (app_standalone.py): hash do código-fonte embutido no bundle;
- clone local: `git rev-parse`.
"""

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

VERSION = "2.0.0"


def _bundle() -> dict | None:
    """Metadados que o app_standalone.py coloca no pacote ao carregá-lo."""
    return getattr(sys.modules.get("appai"), "__bundle__", None)


def _git_sha() -> str:
    here = globals().get("__file__")
    if not here:
        return ""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(here).resolve().parent.parent,
            capture_output=True, text=True, timeout=2, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


@lru_cache(maxsize=1)
def build_info() -> dict:
    sha = os.environ.get("GIT_SHA", "").strip()
    date = os.environ.get("BUILD_DATE", "").strip()
    bundle = _bundle()
    if sha:
        source, commit = "container", sha[:7]
    elif bundle:
        source, commit = "arquivo único", f"src {bundle['digest']}"
    else:
        sha = _git_sha()
        source, commit = "local", sha[:7] if sha else "desconhecido"
    return {
        "version": os.environ.get("APP_VERSION", "").strip() or VERSION,
        "commit": commit,
        "build_date": date,
        "source": source,
    }

"""Histórico de execuções em SQLite.

Local: ./data/appai.db. Container/Kubernetes: APPAI_DATA_DIR=/data (PVC).
Se o diretório não for gravável, o histórico fica desabilitado e o app segue.
"""

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

KINDS = {"sweep": "Benchmark", "coldstart": "Cold start", "rag": "RAG"}


def data_dir() -> Path:
    return Path(os.environ.get("APPAI_DATA_DIR") or Path.cwd() / "data")


class Store:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else data_dir() / "appai.db"
        self.error = ""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._conn() as c:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS runs (
                        id TEXT PRIMARY KEY,
                        created_at REAL NOT NULL,
                        kind TEXT NOT NULL,
                        label TEXT NOT NULL DEFAULT '',
                        endpoint TEXT,
                        model TEXT,
                        info TEXT,
                        data TEXT
                    )""")
                c.execute("CREATE INDEX IF NOT EXISTS runs_created ON runs(created_at)")
            self.enabled = True
        except (OSError, sqlite3.Error) as e:
            self.enabled = False
            self.error = str(e)

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def save(self, kind: str, label: str, info: dict, data: dict) -> str | None:
        if not self.enabled:
            return None
        rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        with self._conn() as c:
            c.execute(
                "INSERT INTO runs (id, created_at, kind, label, endpoint, model, info, data)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (rid, time.time(), kind, label or "", info.get("endpoint"),
                 info.get("model"), json.dumps(info, default=str),
                 json.dumps(data, default=str)),
            )
        return rid

    def list(self, kind: str | None = None, limit: int = 200) -> list:
        if not self.enabled:
            return []
        q = "SELECT id, created_at, kind, label, endpoint, model, info FROM runs"
        args: tuple = ()
        if kind:
            q += " WHERE kind = ?"
            args = (kind,)
        q += " ORDER BY created_at DESC LIMIT ?"
        with self._conn() as c:
            rows = c.execute(q, args + (limit,)).fetchall()
        return [{**dict(r), "info": json.loads(r["info"] or "{}")} for r in rows]

    def get(self, rid: str) -> dict | None:
        if not self.enabled:
            return None
        with self._conn() as c:
            r = c.execute("SELECT * FROM runs WHERE id = ?", (rid,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["info"] = json.loads(d["info"] or "{}")
        d["data"] = json.loads(d["data"] or "{}")
        return d

    def rename(self, rid: str, label: str):
        if self.enabled:
            with self._conn() as c:
                c.execute("UPDATE runs SET label = ? WHERE id = ?", (label, rid))

    def delete(self, rid: str):
        if self.enabled:
            with self._conn() as c:
                c.execute("DELETE FROM runs WHERE id = ?", (rid,))


def display_name(run: dict) -> str:
    when = time.strftime("%d/%m %H:%M", time.localtime(run["created_at"]))
    label = run.get("label") or run.get("model") or run["id"]
    return f"{label} · {KINDS.get(run['kind'], run['kind'])} · {when}"

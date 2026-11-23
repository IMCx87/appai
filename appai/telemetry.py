"""Telemetria de infraestrutura durante o benchmark.

Duas fontes, ambas opcionais:
- **vLLM /metrics** (formato Prometheus), raspado a cada N segundos enquanto o
  benchmark roda: fila, requisições em execução, uso de KV cache, preempções e
  prefix cache.
- **Prometheus HTTP API** (`query_range`) consultado ao final, para DCGM
  (utilização, memória e potência de GPU) ou qualquer PromQL.

Princípio: correlação temporal NÃO é causalidade. O diagnóstico descreve o que
foi observado na mesma janela e usa linguagem cautelosa.
"""

import re
import statistics
import threading
import time

import requests

# Nomes mudaram entre versões do vLLM; o primeiro encontrado vence.
VLLM_GAUGES = {
    "running": ["vllm:num_requests_running"],
    "waiting": ["vllm:num_requests_waiting"],
    "kv": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
}
VLLM_COUNTERS = {
    "preempt": ["vllm:num_preemptions_total", "vllm:num_preemptions"],
    "prefix_hits": ["vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"],
    "prefix_queries": ["vllm:prefix_cache_queries_total",
                       "vllm:gpu_prefix_cache_queries_total"],
}

DEFAULT_PROM_QUERIES = (
    "GPU util (%) | avg(DCGM_FI_DEV_GPU_UTIL)\n"
    "GPU mem (GiB) | sum(DCGM_FI_DEV_FB_USED) / 1024\n"
    "GPU power (W) | sum(DCGM_FI_DEV_POWER_USAGE)"
)

_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)')
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prom_text(text: str) -> dict:
    """Formato de exposição Prometheus → {nome: [(labels, valor), ...]}."""
    out: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, labels_raw, value = m.groups()
        try:
            v = float(value)
        except ValueError:
            continue
        labels = dict(_LABEL.findall(labels_raw or ""))
        out.setdefault(name, []).append((labels, v))
    return out


def _pick(parsed: dict, names: list, agg) -> float | None:
    for n in names:
        if n in parsed:
            return agg([v for _, v in parsed[n]])
    return None


def extract_vllm(parsed: dict) -> dict:
    """Soma entre réplicas/engines; para KV cache usa o máximo (o gargalo)."""
    s = {}
    for key, names in VLLM_GAUGES.items():
        s[key] = _pick(parsed, names, max if key == "kv" else sum)
    for key, names in VLLM_COUNTERS.items():
        s[key] = _pick(parsed, names, sum)
    return s


def scrape_vllm(url: str, verify: bool, headers: dict | None = None,
                timeout: float = 5) -> dict:
    r = requests.get(url, timeout=timeout, verify=verify, headers=headers or {})
    r.raise_for_status()
    return extract_vllm(parse_prom_text(r.text))


class VllmSampler:
    """Raspa o /metrics em uma thread enquanto o benchmark roda."""

    def __init__(self, url: str, interval: float = 1.0, verify: bool = False,
                 headers: dict | None = None):
        self.url, self.interval, self.verify = url, max(interval, 0.2), verify
        self.headers = headers or {}
        self.samples: list = []
        self.errors = 0
        self.last_error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self):
        while not self._stop.is_set():
            t = time.time()
            try:
                s = scrape_vllm(self.url, self.verify, self.headers)
                s["t"] = t
                self.samples.append(s)
            except Exception as e:  # noqa: BLE001
                self.errors += 1
                self.last_error = str(e)[:200]
            self._stop.wait(max(self.interval - (time.time() - t), 0.05))

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> list:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 6)
        return self.samples


def parse_prom_queries(text: str) -> list:
    """Linhas "Nome | PromQL" (ou só PromQL) → [(nome, query)]."""
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            name, q = line.split("|", 1)
            out.append((name.strip(), q.strip()))
        else:
            out.append((line, line))
    return out


def prom_query_range(base: str, query: str, start: float, end: float, step: float,
                     verify: bool, headers: dict | None = None) -> list:
    """Devolve [(t, valor)] — média entre séries, se a query trouxer várias."""
    r = requests.get(
        base.rstrip("/") + "/api/v1/query_range",
        params={"query": query, "start": f"{start:.3f}", "end": f"{end:.3f}",
                "step": f"{max(step, 1):.0f}s"},
        timeout=15, verify=verify, headers=headers or {},
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != "success":
        raise RuntimeError(payload.get("error", "query falhou"))
    by_t: dict = {}
    for series in payload.get("data", {}).get("result", []):
        for t, v in series.get("values", []):
            try:
                by_t.setdefault(float(t), []).append(float(v))
            except ValueError:
                continue
    return [(t, statistics.fmean(vs)) for t, vs in sorted(by_t.items())]


def collect_prometheus(base: str, queries: list, start: float, end: float,
                       step: float, verify: bool, headers: dict | None = None) -> dict:
    series, errors = {}, {}
    for name, q in queries:
        try:
            series[name] = prom_query_range(base, q, start - step, end + step, step,
                                            verify, headers)
        except Exception as e:  # noqa: BLE001
            errors[name] = str(e)[:200]
    return {"series": series, "errors": errors}


# ── Agregação por janela de nível ─────────────────────────────────────────────
def _in_window(points, t0, t1):
    return [p for p in points if t0 <= p[0] < t1]


def aggregate_levels(levels: list, vllm_samples: list, prom_series: dict) -> list:
    """Para cada nível (janela t0–t1), resume a telemetria observada."""
    out = []
    for lv in levels:
        t0, t1 = lv["t0"], lv["t1"]
        agg: dict = {"load": lv["load"]}
        win = [s for s in vllm_samples if t0 <= s["t"] < t1]
        if win:
            for key in ("waiting", "running", "kv"):
                vals = [s[key] for s in win if s.get(key) is not None]
                if vals:
                    agg[f"{key}_max"] = max(vals)
                    agg[f"{key}_mean"] = statistics.fmean(vals)
            # contadores: delta entre a amostra antes/depois da janela
            before = [s for s in vllm_samples if s["t"] <= t0] or win[:1]
            after = [s for s in vllm_samples if s["t"] >= t1] or win[-1:]
            for key in ("preempt", "prefix_hits", "prefix_queries"):
                a, b = before[-1].get(key), after[0].get(key)
                if a is not None and b is not None:
                    agg[f"{key}_delta"] = max(b - a, 0)
            q = agg.get("prefix_queries_delta")
            if q:
                agg["prefix_hit_rate"] = agg.get("prefix_hits_delta", 0) / q
        for name, pts in prom_series.items():
            vals = [v for _, v in _in_window(pts, t0, t1)]
            if vals:
                agg[f"prom:{name}:mean"] = statistics.fmean(vals)
                agg[f"prom:{name}:max"] = max(vals)
        out.append(agg)
    return out


def find_power_series(prom_series: dict) -> list | None:
    for name, pts in prom_series.items():
        low = name.lower()
        if "power" in low or "potência" in low or "(w)" in low:
            return pts
    return None


def energy_kwh(power_points: list, t0: float, t1: float) -> float | None:
    """Integra potência (W) no tempo (trapézios) → kWh na janela."""
    pts = [p for p in power_points or [] if t0 <= p[0] <= t1]
    if len(pts) < 2:
        if pts:
            return pts[0][1] * (t1 - t0) / 3.6e6
        return None
    joules = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(pts, pts[1:]))
    return joules / 3.6e6


# ── Diagnóstico com linguagem cautelosa ───────────────────────────────────────
def _prom_val(agg: dict, needle: str, stat: str = "mean"):
    for k, v in agg.items():
        if k.startswith("prom:") and needle in k.lower() and k.endswith(":" + stat):
            return v
    return None


def diagnose(rows: list, infra: list, unit: str = "usuários simultâneos") -> list:
    """Achados observacionais por nível. Cada item: {level, kind, text}.
    Nunca afirma causa: descreve o que apareceu na mesma janela."""
    by_load = {a["load"]: a for a in infra}
    findings = []
    prev_row = None
    first_fail_done = False
    for row in sorted(rows, key=lambda r: r["load"]):
        a = by_load.get(row["load"], {})
        L = row["load"]
        kv = a.get("kv_max")
        waiting = a.get("waiting_max")
        util = _prom_val(a, "util")
        if not row["slo_pass"] and not first_fail_done:
            first_fail_done = True
            parts = [f"Em {L:g} {unit} o SLO deixou de ser cumprido "
                     f"(TTFT p95 {row['ttft_p95'] or 0:.0f} ms, ITL p95 "
                     f"{row['itl_p95'] or 0:.1f} ms, erros {row['err_pct']:.1f}%)."]
            if waiting and waiting >= 1:
                parts.append(f"Na mesma janela havia até {waiting:.0f} requisição(ões) "
                             "na fila do servidor, o que pode estar relacionado ao "
                             "aumento do TTFT (o batch máximo do vLLM — `max_num_seqs` "
                             "— ou a memória de KV cache podem ter sido atingidos).")
            if util is not None and util < 50:
                parts.append(f"A GPU teve utilização média de {util:.0f}%: o gargalo "
                             "pode estar fora da GPU (rede, CPU, gateway/LB, limite "
                             "de concorrência).")
            findings.append({"level": L, "kind": "slo", "text": " ".join(parts)})
        if kv is not None and kv >= 0.9:
            findings.append({"level": L, "kind": "kv", "text":
                f"Em {L:g} {unit} o KV cache chegou a {kv * 100:.0f}%. Com o cache "
                "cheio, o vLLM passa a enfileirar ou preemptar requisições; mais "
                "memória de GPU, KV cache em FP8 ou `max_model_len` menor podem "
                "aumentar a concorrência suportada."})
        pre = a.get("preempt_delta")
        if pre:
            findings.append({"level": L, "kind": "preempt", "text":
                f"Em {L:g} {unit} houve {pre:.0f} preempção(ões): requisições foram "
                "interrompidas e recomputadas por falta de KV cache — sinal de "
                "pressão de memória que costuma elevar a latência de cauda."})
        if prev_row and prev_row.get("throughput") and row.get("throughput"):
            gain = row["throughput"] / prev_row["throughput"] - 1
            load_gain = L / prev_row["load"] - 1 if prev_row["load"] else 0
            if load_gain >= 0.5 and gain < 0.1:
                txt = (f"De {prev_row['load']:g} para {L:g} {unit} a vazão agregada "
                       f"variou só {gain * 100:+.0f}%: o endpoint parece ter atingido "
                       "a saturação; carga extra vira fila e latência.")
                if util is not None and util >= 85:
                    txt += f" A GPU estava em {util:.0f}% de utilização média."
                findings.append({"level": L, "kind": "plateau", "text": txt})
        hit = a.get("prefix_hit_rate")
        if hit is not None and hit >= 0.3:
            findings.append({"level": L, "kind": "prefix", "text":
                f"Em {L:g} {unit} {hit * 100:.0f}% dos blocos de prompt vieram do "
                "prefix cache: prompts com início em comum reduzem o custo de "
                "prefill e o TTFT."})
        if row.get("lag_p95") and row["lag_p95"] > 250:
            findings.append({"level": L, "kind": "client", "text":
                f"Em {L:g} {unit} o próprio gerador atrasou os envios (p95 "
                f"{row['lag_p95']:.0f} ms). Os números desse nível podem estar "
                "limitados pelo cliente, não pelo endpoint."})
        prev_row = row
    return findings

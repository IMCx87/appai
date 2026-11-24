# AI Infra Benchmark (NAI Endpoint Chat)

A Streamlit chat client **and benchmark tool** for OpenAI-compatible inference endpoints, built for testing and demonstrating **Nutanix Enterprise AI (NAI)** and **vLLM** deployments in front of customers.

Generic LLM benchmarks report numbers. This one is built to answer the questions an infrastructure conversation actually asks, and to explain the answer from the infrastructure side:

- **How many concurrent users does this endpoint serve within the SLO?** (capacity and goodput, not just peak tokens/s)
- **Why does latency degrade at that point?** (queue, KV cache, preemptions and GPU utilization observed in the same time window)
- **How long does a hibernated or scaled-to-zero model take to come back, and how much of that is storage?**
- **Where does the time go in a RAG answer?** (embedding → vector search → prefill → decode)
- **What would the same traffic cost on a public cloud, and where is the break-even?**

It runs three ways from the same code: as a **single file** you copy anywhere (`streamlit run app_standalone.py`), from a **clone** of this repository, or as a **container** on Kubernetes/NKP with GitOps. See [Ways to run](#ways-to-run).

## From zero to "how many users?" in five minutes

No GPU, no cluster, no clone. One file, three commands:

```bash
curl -O https://raw.githubusercontent.com/IMCx87/appai/main/app_standalone.py
pip install streamlit requests plotly
streamlit run app_standalone.py --theme.base dark
```

Then a few clicks. In the sidebar, open *🧪 Mock local* and click **Usar mock embutido**: a simulated vLLM/NAI endpoint starts inside the app and every URL is filled in. In **📊 Benchmark**, add `16` and `32` to the concurrency levels and click **▶ Executar benchmark**.

About a minute and a half later the screen shows (the UI is in Portuguese):

> 🎯 **Capacidade dentro do SLO: 16** usuários simultâneos (falhou em 32)
>
> Em 32 usuários simultâneos o SLO deixou de ser cumprido (TTFT p95 ≈ 5800 ms …). Na mesma janela havia até 16 requisição(ões) na fila do servidor, o que pode estar relacionado ao aumento do TTFT…
>
> De 16 para 32 usuários simultâneos a vazão agregada variou só +0%: o endpoint parece ter atingido a saturação; carga extra vira fila e latência.

In plain words: up to 16 users get a first token in about 0.2 s; at 32, the extra users wait in line and the first token takes almost 6 s, while total throughput doesn't grow at all. That is the whole point of the tool: a number the customer understands, and the infrastructure reason behind it. Now swap the mock for a real endpoint and do it again.

## Features

| Tab | What it does |
|---|---|
| 💬 **Chat** | SSE streaming (with non-streaming fallback), per-response TTFT, ITL, tokens/s, token counts, estimated cloud cost; session KPI strip |
| 📊 **Benchmark** | Closed-loop (fixed concurrency) **or open-loop (Poisson arrival rate)** sweeps; fixed or **lognormal** prompt/output sizes; **real-prompt datasets** (JSONL, ShareGPT, TXT); prefix-cache bypass; forced output length (`ignore_eos`); warm-up. **SLO** (TTFT p95, ITL p95, error rate) → **capacity within SLO and goodput**, recomputed live when the SLO changes. Optional **infra telemetry**: vLLM `/metrics` sampled during the run plus Prometheus/DCGM queries, overlaid per level, with **hedged automatic findings**. CSV / JSON / **standalone HTML report** |
| 🧊 **Cold start** | Measures time-to-ready and time-to-first-token of a hibernated / scaled-to-zero endpoint, labeled by **storage backend** (Files, Volumes, Objects, local NVMe), with an effective load-throughput estimate |
| 📚 **RAG** | Indexes a corpus (built-in sample or your `.txt`/`.md`) via `/v1/embeddings`, then measures each stage of the pipeline under concurrency, including the **TTFT the user actually perceives** |
| 🗂️ **History & A/B** | Every run is saved (SQLite). Compare runs side by side with deltas against a baseline (e.g. NAI vs. cloud API, L40S vs. H100, FP16 vs. quantized, Files vs. Volumes). **Live side-by-side**: the same prompt to two endpoints at once |
| 💰 **Cost** | Cloud price references (editable), cost per exchange, and **on-prem vs. cloud** using the throughput measured *within the SLO* (or peak), average utilization, input+output volumes, optional **energy** (power, PUE, $/kWh; measured power from DCGM when available), replicas needed, break-even and a cost-vs-volume chart |

Also: exponential-backoff retries including NAI's `404 hibernated endpoint`, Bearer / `x-api-key` / no-auth, `/v1/models` discovery, debug panel with redacted credentials, and a **built-in mock server** so you can rehearse a demo with no GPU.

> The benchmark, cold-start and RAG tests send real requests: they consume tokens and GPU time on the target endpoint. Prices, GPU-hour cost, utilization and energy are **editable assumptions, not measurements**.

## Quick start

Requires Python 3.10+ (tested on 3.11 and 3.13; CI and the image use 3.11). The fastest path is the [single file](#single-file); from a clone:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # streamlit, requests, plotly
streamlit run app.py
```

Open `http://localhost:8501`, set the endpoint URL and API key in the sidebar, click **Conectar** to list models, then use the tabs.

**No GPU nearby?** Open *🧪 Mock local (sem GPU)* in the sidebar and click **Usar mock embutido**. It starts an in-process server that imitates vLLM/NAI (streaming, batch limit with queueing, finite KV cache with preemptions, prefix cache, hibernation, embeddings and synthetic DCGM metrics) and fills in every URL. The numbers are plausible, not real. You can also run it standalone:

```bash
python -m appai.mock_server --port 8000 --max-batch 16 --wake-seconds 20
# endpoint http://localhost:8000/v1 · metrics /metrics · Prometheus API at the root
# hibernate it with: curl -X POST localhost:8000/admin/hibernate
```

## Ways to run

| Mode | Command | Good for |
|---|---|---|
| Single file | `streamlit run app_standalone.py` | Copying to any machine with Python: laptop, jump host, a VM in the customer's lab |
| Repository clone | `streamlit run app.py` | Development |
| Container / Kubernetes | `docker run …` / `kubectl apply -k k8s/` | Shared deployment on NKP, GitOps |

All three run the same code and show where they came from in the sidebar footer: `arquivo único` + source hash, `local` + git commit, or `container` + the commit and build date baked into the image.

### Single file

`app_standalone.py` is the **whole app in one file**: copy it anywhere (a laptop, a jump host, a VM in the customer's lab) and run it with Streamlit. No clone, no package, same features: chat, benchmark, telemetry, cold start, RAG, history/A-B, cost and the built-in mock.

```bash
pip install streamlit requests plotly
streamlit run app_standalone.py --theme.base dark
python app_standalone.py mock --port 8000      # mock vLLM/NAI server, no Streamlit needed
```

History goes to `./data/appai.db` next to where you run it (`APPAI_DATA_DIR` changes that). The sidebar shows `arquivo único` and a source hash, so you know which build a copy came from.

The file is **generated** from `appai/` + `app.py` by `tools/build_standalone.py`. It embeds each module's source and registers an importer that serves them, so the single file runs exactly the same code as the package; nothing is rewritten by hand. Don't edit it: change the package and regenerate. CI fails if the committed copy is out of date.

## Running it with a customer

Each step answers a question the customer is already asking.

1. **"Is it fast?"** · *💬 Chat.* Ask one question. Every answer shows time to first token, tokens/s and what it would have cost on a public cloud. Ten seconds, and there is a baseline on the screen.
2. **"How many people can use it?"** · *📊 Benchmark.* Agree on the SLO **with them** before running: for example, first token under 1 s and fluid reading under 50 ms per token. The number that comes out is theirs, not a vendor claim. Lead with *capacity within SLO*, not peak tokens/s.
3. **"Why does it stop there?"** · *🔎 O que a infraestrutura mostrou* and the telemetry chart. Queue, KV cache and GPU side by side with latency. This is where the infrastructure conversation starts: more GPU memory, FP8 KV cache, another replica.
4. **"And when it scales to zero?"** · *🧊 Cold start*, once per storage backend. Their storage, measured in seconds.
5. **"What about our documents?"** · *📚 RAG.* Where the time goes in an answer with retrieval, including the wait the user actually feels.
6. **"Is it worth it versus the cloud?"** · *💰 Custo*, using throughput *within the SLO* and their volumes and prices. Break-even on one chart.
7. **Leave something behind.** Label each run (GPU, model, storage), line them up in *🗂️ Histórico & A/B*, and send the HTML report. It opens offline and prints to PDF.

Rehearse the whole script on the mock first. It streams, queues, hibernates and reports KV-cache and GPU metrics, so you will know every click before anyone is watching. Its numbers are plausible, not real.

## How to read the results

**SLO, capacity and goodput.** A level *passes* when TTFT p95 and ITL p95 are within target and the error rate is under the limit. *Capacity* is the highest tested load at which that level and all lower levels pass. *Goodput* counts only the requests that individually met the SLO. Peak throughput is still shown, but it is usually reached when latency is already unacceptable, so it overstates what the endpoint can deliver with quality.

**Closed-loop vs. open-loop.** Closed-loop simulates N users who wait for each answer before asking again, so load backs off by itself when the server slows down. Open-loop sends requests at a Poisson rate regardless of how the server is doing, as real traffic does. Beyond saturation the queue grows without bound and TTFT explodes. Use open-loop to find a sustainable request rate; use closed-loop to size concurrent users.

**Infra findings.** With telemetry enabled, each level's window is matched against queue depth, running requests, KV-cache usage, preemptions, prefix-cache hit rate and Prometheus series (GPU util, memory, power). Findings describe what was **observed in the same window** ("may be related to…") and never claim causation. The generator also reports its own send lag, so a client-limited level is flagged instead of blamed on the endpoint.

**Telemetry on NKP / NAI.** The vLLM `/metrics` endpoint lives on the model's serving pod. If the app runs in the same cluster, use the service DNS name (`http://<service>.<namespace>:8000/metrics`). From outside, use `kubectl port-forward` or point to a Prometheus that already scrapes vLLM. The default PromQL queries use the DCGM exporter installed by the NVIDIA GPU Operator. Any `Name | PromQL` line works; put `(W)` in the power series name to enable the energy calculation.

**Cold start.** Hibernate the endpoint (NAI UI, `minReplicas: 0`, or `kubectl scale --replicas=0`), click *Medir cold start*, and repeat per storage backend. Time-to-ready includes pod scheduling, image pull and runtime init, so the derived GB/s is a lower bound, not the storage throughput. A first cold start after a node reboot (cold page cache) is the honest worst case.

**RAG.** Vector search runs in memory (numpy, brute-force cosine). It measures the pipeline and the embedding endpoint, not a production vector database.

## Configuration

| Setting | Where | Notes |
|---|---|---|
| API Endpoint URL / API Key / Auth / Model | Sidebar | URL with or without `/v1`; the key stays in session state, never on disk |
| Temperature / Max tokens / Retries | Sidebar expander | Retries cover 429, 5xx and hibernated 404 |
| TLS verification | Sidebar toggle | Off by default for lab environments with self-signed certs |
| SLO, workload, telemetry URLs | **Benchmark** tab | Changing the SLO recomputes capacity without re-running |
| Cloud prices, GPU cost, utilization, energy | **Custo** tab | Illustrative defaults: adjust to the scenario |

Environment variables (all optional) pre-fill the UI: `NAI_API_URL`, `NAI_API_KEY`, `NAI_MODEL`, `NAI_EMBED_MODEL`, `VLLM_METRICS_URL`, `PROMETHEUS_URL`. `APPAI_DATA_DIR` sets where the run history (`appai.db`) is stored: `./data` locally, `/data` in the container. If the directory is not writable, history is disabled and the app keeps working.

The sidebar footer shows the app version, how it is running (single file, local clone or container), the commit or source hash, and where history is stored.

## Docker

`.github/workflows/docker-build.yml` runs lint (`ruff`), checks that `app_standalone.py` is in sync with the package, and runs the tests (`pytest`) on every push and pull request. On pushes it then builds a **multi-arch image (`linux/amd64`, `linux/arm64`)** and pushes it to Docker Hub (`imcx87/appai`) with tags for the branch name and `sha-<commit>`; on `main` also `latest` and `main-<timestamp>-<sha>` (sortable, used by Flux image automation). The commit and build date are baked into the image. Pushing requires the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets; without them the job still builds both architectures (so the Dockerfile is validated) and emits a warning instead of pushing. Commits that only touch `k8s/`, `flux/` or Markdown don't trigger a new image.

> **Current status:** the Docker Hub secrets are not configured in this repository yet, so CI builds the image but does not publish it (and no earlier run published one either). Add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` under *Settings → Secrets and variables → Actions* before deploying to Kubernetes; the manifests pull `imcx87/appai` from Docker Hub.

The container runs as non-root (UID 10001), exposes port `8501` and keeps history in the `/data` volume.

```bash
docker build -t imcx87/appai:local .
docker run -p 8501:8501 -v appai-data:/data imcx87/appai:local
```

## Kubernetes / GitOps (Nutanix Kubernetes Platform)

Prerequisite: a published image (see the status note above). `k8s/` holds plain manifests: Namespace, a 1 Gi **PVC for the run history** (default StorageClass), Deployment (one replica, `Recreate` strategy because the volume is RWO, non-root with `fsGroup`, health probes on `/_stcore/health`, requests/limits), NodePort Service on `30851`, and Kustomization.

```bash
kubectl apply -k k8s/
```

For GitOps via Flux, apply `flux/` on the management/workspace cluster (namespace `flux-system`). Flux tracks `main` and reconciles `k8s/` into the `appai` namespace every 5 minutes:

```bash
kubectl apply -f flux/gitrepository.yaml
kubectl apply -f flux/kustomization.yaml
```

**Automatic image updates (optional).** Without automation the Deployment stays on `imcx87/appai:main`, and a new push only reaches the cluster when the pod restarts. `flux/image-automation/` makes the rollout real GitOps: Flux watches Docker Hub, picks the newest `main-<timestamp>-<sha>` tag and commits it to `k8s/deployment.yaml` (through the `$imagepolicy` marker), so every running version is traceable in Git.

```bash
# 1. image controllers (if not already installed)
flux install --components-extra=image-reflector-controller,image-automation-controller
# 2. deploy key with write access (add the printed public key in GitHub → Settings → Deploy keys)
flux create secret git appai-deploy-key --url=ssh://git@github.com/IMCx87/appai --namespace=flux-system
# 3. automation objects
kubectl apply -k flux/image-automation/
```

## Development

```bash
pip install -r requirements.txt pytest ruff
ruff check .
python tools/build_standalone.py   # regenerate app_standalone.py after changing appai/ or app.py
pytest -q        # unit + integration tests against the mock server, Streamlit AppTest,
                 # and the single-file build (in sync + runs alone from an empty folder)
```

```
├── app.py                    # Streamlit entry point (thin: page config + tabs)
├── app_standalone.py         # GENERATED single-file version (do not edit)
├── appai/
│   ├── client.py             # OpenAI-compatible HTTP client, SSE parsing, retries
│   ├── bench.py              # closed/open-loop sweeps, summaries, SLO, capacity, goodput
│   ├── workload.py           # prompts, size distributions, datasets
│   ├── telemetry.py          # vLLM /metrics sampler, Prometheus queries, findings
│   ├── coldstart.py          # cold-start measurement
│   ├── rag.py                # RAG pipeline with per-stage timing
│   ├── cost.py               # cloud vs. on-prem model
│   ├── store.py              # run history (SQLite)
│   ├── report.py / charts.py # HTML report and shared Plotly figures
│   ├── mock_server.py        # vLLM/NAI imitation for demos and tests
│   ├── version.py            # version / commit / build date
│   └── ui/                   # one module per tab + sidebar + session state
├── tools/build_standalone.py # generates app_standalone.py
├── tests/                    # pytest suite
├── Dockerfile · requirements.txt · pyproject.toml · .streamlit/config.toml
├── .github/workflows/docker-build.yml
├── k8s/                      # Namespace, PVC, Deployment, Service, Kustomization
└── flux/                     # GitRepository + Kustomization; image-automation/ (optional)
```

Everything outside `appai/ui/` is plain Python with no Streamlit dependency, so it can be tested or reused from a script.

## Roadmap

- Programmatic hibernate/resume via the NAI management REST API (today: hibernate in the UI or via a configurable POST URL)
- Production vector databases for the RAG test (pgvector, Milvus)
- Multi-turn conversations in the workload generator, to exercise prefix caching realistically

## License

MIT. Use it, break it, improve it.

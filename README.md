# AI Infra Benchmark (NAI Endpoint Chat)

A Streamlit chat client **and benchmark tool** for OpenAI-compatible inference endpoints, built for testing and demonstrating **Nutanix Enterprise AI (NAI)** and **vLLM** deployments in front of customers.

Generic LLM benchmarks report numbers. This one is built to answer the questions an infrastructure conversation actually asks, and to explain the answer from the infrastructure side:

- **How many concurrent users does this endpoint serve within the SLO?** (capacity and goodput, not just peak tokens/s)
- **Why does latency degrade at that point?** (queue, KV cache, preemptions and GPU utilization observed in the same time window)
- **How long does a hibernated or scaled-to-zero model take to come back, and how much of that is storage?**
- **Where does the time go in a RAG answer?** (embedding → vector search → prefill → decode)
- **What would the same traffic cost on a public cloud, and where is the break-even?**

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

The sidebar footer shows the app version, the commit it was built from, whether it runs from a local clone or a container, and where history is stored.

## Docker

`.github/workflows/docker-build.yml` runs lint (`ruff`) and tests (`pytest`) on every push and pull request. On pushes it then builds a **multi-arch image (`linux/amd64`, `linux/arm64`)** and pushes it to Docker Hub (`imcx87/appai`) with tags for the branch name and `sha-<commit>`; on `main` also `latest` and `main-<timestamp>-<sha>` (sortable, used by Flux image automation). The commit and build date are baked into the image. Requires the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets. Commits that only touch `k8s/`, `flux/` or Markdown don't trigger a new image.

The container runs as non-root (UID 10001), exposes port `8501` and keeps history in the `/data` volume.

```bash
docker build -t imcx87/appai:local .
docker run -p 8501:8501 -v appai-data:/data imcx87/appai:local
```

## Kubernetes / GitOps (Nutanix Kubernetes Platform)

`k8s/` holds plain manifests: Namespace, a 1 Gi **PVC for the run history** (default StorageClass), Deployment (one replica, `Recreate` strategy because the volume is RWO, non-root with `fsGroup`, health probes on `/_stcore/health`, requests/limits), NodePort Service on `30851`, and Kustomization.

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
pytest -q        # unit + integration tests against the mock server, plus Streamlit AppTest
```

```
├── app.py                    # Streamlit entry point (thin: page config + tabs)
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

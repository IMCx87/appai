# AI Infra Benchmark (NAI Endpoint Chat)

A Streamlit chat client **and benchmark tool** for OpenAI-compatible inference endpoints, built for testing and demonstrating **Nutanix Enterprise AI (NAI)** and **vLLM** deployments in front of customers.

It surfaces the numbers that matter when the conversation is about infrastructure: time to first token, inter-token latency, throughput under concurrency, and what the same traffic would cost on a public cloud.

## Features

**Chat**
- SSE streaming with a non-streaming fallback
- Per-response metrics: tokens/s, TTFT, inter-token latency (ITL), prompt/completion tokens, served model, and estimated public-cloud cost
- Session KPI strip: responses, average TTFT and tokens/s, tokens in/out, accumulated cost

**Benchmark**
- Load profiles (short chat, long-prompt RAG, long generation, custom) with adjustable input/output token sizes
- Concurrency sweep (1 → 64 simultaneous requests) with per-level results: TTFT p50/p95/p99, ITL, tokens/s per request, aggregate throughput, req/s, errors and retries
- Charts: throughput × latency curve, TTFT distribution, per-request speed distribution
- Optional unique prompt prefix per request to bypass server-side prefix caching
- Export results as CSV / JSON

**Cost**
- Public-cloud price references (editable, per 1M tokens in/out), cost per message exchange, projection per 1,000 exchanges
- On-prem vs. cloud estimate using the peak throughput measured by the benchmark, a GPU-hour cost and a monthly volume (break-even included)
- Prices and GPU cost are **editable assumptions, not measurements**

**Resilience & access**
- Exponential backoff retry with jitter, including the NAI `404 hibernated endpoint` response from a hibernated target in a Unified Endpoint pool
- Bearer token, `x-api-key`, or no auth; `GET /v1/models` discovery doubles as a connectivity check
- Debug panel with the last request/response, credentials redacted

## Run locally (single file)

Only `app.py` is required (`.streamlit/config.toml` just adds the dark theme).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # streamlit, requests, plotly
streamlit run app.py
```

Open `http://localhost:8501`, set the endpoint URL and API key in the sidebar, click **Conectar** to list models, then use the **Chat**, **Benchmark** and **Custo** tabs.
Initial sidebar values can be set with the `NAI_API_URL` and `NAI_API_KEY` environment variables.

## Configuration

| Setting | Where | Notes |
|---|---|---|
| API Endpoint URL | Sidebar | Base URL with or without `/v1`; the app normalizes it |
| API Key | Sidebar | Kept only in session state, never written to disk |
| Auth Header | Sidebar | `Authorization: Bearer`, `x-api-key`, or none |
| Model | Sidebar | Auto-populated from `/v1/models` after connecting |
| Temperature / Max tokens / Retries | Sidebar expander | Retries cover 429, 5xx and hibernated 404 |
| TLS verification | Sidebar toggle | Off by default for lab environments with self-signed certs |
| Cloud prices, GPU cost | **Custo** tab | Illustrative defaults — adjust to the scenario |

> The benchmark sends real requests: it consumes tokens and GPU time on the target endpoint.

## Docker

The image is built and pushed to Docker Hub (`imcx87/appai`) by `.github/workflows/docker-build.yml` on every push (`linux/amd64`). Tags: the branch name, `sha-<commit>`, and `latest` for `main`. Requires the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets.

The container runs as a non-root user (UID 10001) and exposes port `8501`.

```bash
docker build -t imcx87/appai:local .
docker run -p 8501:8501 imcx87/appai:local
```

## Kubernetes / GitOps (Nutanix Kubernetes Platform)

`k8s/` has plain manifests (Namespace, Deployment, NodePort Service on `30851`, Kustomization). The Deployment runs non-root with a restricted-style `securityContext`, health probes on `/_stcore/health`, and resource requests/limits.

```bash
kubectl apply -k k8s/
```

For GitOps via Flux, apply the resources in `flux/` on the management/workspace cluster (namespace `flux-system`):

```bash
kubectl apply -f flux/gitrepository.yaml
kubectl apply -f flux/kustomization.yaml
```

Flux tracks the `main` branch and reconciles `k8s/` into the `appai` namespace every 5 minutes. There is no image-update automation: bump the image tag in `k8s/deployment.yaml` (for example to `sha-<commit>`) and push to roll out a new version. To point the app at an endpoint by default, uncomment the `NAI_API_URL` env var in the Deployment.

## Project structure

```
├── app.py                  # Streamlit application (single file)
├── requirements.txt        # streamlit, requests, plotly
├── Dockerfile
├── .streamlit/config.toml  # Dark theme (optional)
├── .github/workflows/
│   └── docker-build.yml    # Build & push to Docker Hub
├── k8s/                    # Namespace, Deployment, Service, Kustomization
└── flux/                   # Flux GitRepository + Kustomization
```

## Roadmap

- Side-by-side comparison mode: two endpoints answering the same prompt with parallel metrics
- Programmatic hibernate/resume via the NAI management REST API

## License

MIT. Use it, break it, improve it.

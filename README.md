# NAI Endpoint Chat

A lightweight Streamlit chat client for OpenAI-compatible inference endpoints, built for testing and demonstrating **Nutanix Enterprise AI (NAI)** and **vLLM** deployments in front of customers.

Beyond basic chat, it surfaces the numbers that matter when the conversation is about infrastructure: time to first token, throughput, and token accounting per response.

## Features

- **SSE streaming**: responses render token by token, with a non-streaming fallback toggle
- **Live inference metrics per response**: tokens/s, TTFT (ms), prompt/completion tokens, and the model that actually served the request
- **Endpoint discovery**: connects to `GET /v1/models` and populates a model selector, doubling as a connectivity and auth check
- **Resilient by default**: exponential backoff retry with jitter for transient failures, including the NAI `404 hibernated endpoint` response returned when a load-balanced target in a Unified Endpoint pool is hibernated
- **Flexible auth**: Bearer token, `x-api-key`, or no auth
- **Generation controls**: temperature, max tokens, and system prompt in the sidebar
- **Debug panel**: full request/response log of the last call, with credentials redacted
- **Native dark theme** via `.streamlit/config.toml`, no injected CSS

## Requirements

- Python 3.10+
- An OpenAI-compatible endpoint (NAI inference endpoint, NAI Unified Endpoint, or raw vLLM)

## Quick start

```bash
git clone <this-repo>
cd nai-endpoint-chat

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`, set the endpoint URL and API key in the sidebar, click **Conectar** to list available models, and start chatting.

## Configuration

| Setting | Where | Notes |
|---|---|---|
| API Endpoint URL | Sidebar | Base URL with or without `/v1`; the app normalizes it |
| API Key | Sidebar | Stored only in session state, never written to disk |
| Auth Header | Sidebar | `Authorization: Bearer`, `x-api-key`, or none |
| Model | Sidebar | Auto-populated from `/v1/models` after connecting |
| Temperature / Max tokens | Sidebar expander | Passed through to the inference request |
| Retries | Sidebar expander | Attempts for transient errors (429, 5xx, hibernated 404) |
| TLS verification | Sidebar toggle | Off by default for lab environments with self-signed certs |

## About the hibernated endpoint retry

NAI supports hibernating inference endpoints to release GPU and compute resources while retaining configuration. When a load balancer or Unified Endpoint routes a request to a hibernated target, the gateway returns `404 {"message": "Inference request failed, due to hibernated endpoint"}`.

This client treats that specific response as transient and retries with exponential backoff, giving the balancer a chance to route the next attempt to a healthy target in the pool. If all targets are hibernated, retries are exhausted and the error is surfaced with full context in the debug panel.

The proper long-term fix belongs at the gateway layer (health-based target ejection or failover on hibernated state). The client-side retry is defense in depth.

## Project structure

```
nai-endpoint-chat/
├── app.py                  # Streamlit application
├── requirements.txt        # streamlit + requests, nothing else
├── .streamlit/
│   └── config.toml         # Dark theme
├── Dockerfile
├── .github/workflows/
│   └── docker-build.yml    # Builds and pushes the image to Docker Hub on every push
├── k8s/                    # Plain manifests (Namespace, Deployment, Service, Kustomization)
├── flux/                   # Flux GitRepository + Kustomization for GitOps deployment
└── .gitignore
```

## Docker

The image is built and pushed to Docker Hub (`imcx87/appai`) automatically by
`.github/workflows/docker-build.yml` on every push, tagged with the branch name
(plus `latest` for `main`). Requires the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`
repository secrets.

Build locally:

```bash
docker build -t imcx87/appai:local .
docker run -p 8501:8501 imcx87/appai:local
```

## Kubernetes / GitOps (Nutanix Kubernetes Platform)

`k8s/` has plain manifests deployable with:

```bash
kubectl apply -k k8s/
```

For GitOps via NKP's built-in Flux, apply the resources in `flux/` on the
management/workspace cluster (namespace `flux-system`):

```bash
kubectl apply -f flux/gitrepository.yaml
kubectl apply -f flux/kustomization.yaml
```

This makes Flux track the `main` branch of this repo and reconcile `k8s/`
into the `appai` namespace every 5 minutes. There's no image-update automation
configured — bumping the image tag in `k8s/deployment.yaml` and pushing is
what triggers a new rollout. The Service is `NodePort` on `30851`.

## Roadmap

- Side-by-side comparison mode: two endpoints answering the same prompt with parallel metrics
- Programmatic hibernate/resume via the NAI management REST API
- Latency distribution chart across a session

## License

MIT. Use it, break it, improve it.

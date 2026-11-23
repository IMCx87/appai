FROM python:3.11-slim

# Versão/commit exibidos na sidebar e gravados em cada execução (preenchidos pelo CI)
ARG APP_VERSION=""
ARG GIT_SHA=""
ARG BUILD_DATE=""

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    APP_VERSION=${APP_VERSION} \
    GIT_SHA=${GIT_SHA} \
    BUILD_DATE=${BUILD_DATE} \
    APPAI_DATA_DIR=/data

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY appai/ appai/
COPY .streamlit/ .streamlit/

# UID numérico não-root: compatível com runAsNonRoot / Pod Security "restricted".
# /data guarda o histórico (SQLite); no Kubernetes é um PVC.
RUN useradd --uid 10001 --create-home appuser \
    && mkdir -p /data && chown -R appuser /app /data
USER 10001
VOLUME ["/data"]

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

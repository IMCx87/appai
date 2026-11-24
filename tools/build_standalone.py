"""Gera app_standalone.py: o app inteiro em um único arquivo para `streamlit run`.

    python tools/build_standalone.py          # (re)gera
    python tools/build_standalone.py --check  # falha se estiver desatualizado (CI)

Como funciona: o código de cada módulo do pacote `appai/` é embutido como
texto, e um importador registrado em sys.meta_path entrega esses módulos
quando o app faz `import appai...`. Nada é reescrito, então o arquivo único
se comporta exatamente como o pacote — e a fonte da verdade continua sendo
`appai/` + `app.py`. Não edite o arquivo gerado.
"""

import ast
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "appai"
OUT = ROOT / "app_standalone.py"

HEADER = '''"""
AI Infra Benchmark — versão ARQUIVO ÚNICO (gerada automaticamente)

Chat + benchmark para endpoints OpenAI-compatible (vLLM / Nutanix Enterprise AI)
com SLO, telemetria, cold start, RAG, histórico/A-B e custo — o mesmo app do
repositório, em um só arquivo para copiar e rodar onde quiser.

    pip install streamlit requests plotly
    streamlit run app_standalone.py

Mock vLLM/NAI sem GPU: botão na sidebar, ou
    python app_standalone.py mock --port 8000

Histórico: ./data/appai.db (mude com APPAI_DATA_DIR).
Tema escuro (opcional): streamlit run app_standalone.py --theme.base dark
(ou o .streamlit/config.toml do repositório ao lado do arquivo).

NÃO EDITE: gerado por tools/build_standalone.py a partir de appai/ e app.py
(fonte {digest}). Altere o pacote e rode o gerador de novo.
"""
# ruff: noqa
# fmt: off

import importlib.abc
import importlib.util
import linecache
import sys

_SOURCES = {{
{sources}
}}
_PACKAGES = {packages}
_DIGEST = "{digest}"


class _AppaiBundleImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Entrega os módulos appai.* a partir de _SOURCES."""

    _appai_bundle = True

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _SOURCES:
            return None
        return importlib.util.spec_from_loader(fullname, self,
                                               is_package=fullname in _PACKAGES)

    def create_module(self, spec):
        return None

    def get_source(self, fullname):
        return _SOURCES[fullname]

    def exec_module(self, module):
        name = module.__name__
        src = _SOURCES[name]
        # nome "virtual" só para tracebacks legíveis (linecache tem o código)
        fn = "<app_standalone>/" + name.replace(".", "/") + (
            "/__init__.py" if name in _PACKAGES else ".py")
        linecache.cache[fn] = (len(src), None, src.splitlines(True), fn)
        if name == "appai":
            module.__bundle__ = {{"digest": _DIGEST}}
        exec(compile(src, fn, "exec"), module.__dict__)


# O Streamlit reexecuta este arquivo a cada interação: instala uma vez só.
if not any(getattr(f, "_appai_bundle", False) for f in sys.meta_path):
    sys.meta_path.insert(0, _AppaiBundleImporter())


def _inside_streamlit() -> bool:
    try:
        from streamlit import runtime
        return runtime.exists()
    except Exception:
        return False


if __name__ == "__main__" and sys.argv[1:2] == ["mock"] and not _inside_streamlit():
    from appai.mock_server import main as _mock_main
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    _mock_main()
    sys.exit(0)

# ── app.py ────────────────────────────────────────────────────────────────────
{app}'''


def module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def collect() -> tuple[dict, list]:
    files = sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)
    sources = {module_name(p): p.read_text(encoding="utf-8") for p in files}
    packages = sorted(module_name(p) for p in files if p.name == "__init__.py")
    return sources, packages


def strip_docstring(src: str) -> str:
    """Remove o docstring do app.py (o cabeçalho do bundle substitui)."""
    tree = ast.parse(src)
    first = tree.body[0] if tree.body else None
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
            and isinstance(first.value.value, str):
        lines = src.splitlines(True)
        return "".join(lines[first.end_lineno:]).lstrip("\n")
    return src


def literal(src: str) -> str:
    # r'''...''' mantém o código legível no arquivo gerado
    if "'''" in src or src.endswith("\\"):
        return repr(src)
    return "r'''" + src + "'''"


def build() -> str:
    sources, packages = collect()
    app_src = (ROOT / "app.py").read_text(encoding="utf-8")
    h = hashlib.sha256()
    for name in sorted(sources):
        h.update(name.encode() + b"\0" + sources[name].encode() + b"\0")
    h.update(app_src.encode())
    digest = h.hexdigest()[:10]
    body = ",\n".join(f"    {name!r}: {literal(src)}" for name, src in sources.items())
    pkgs = "{" + ", ".join(repr(x) for x in packages) + "}"  # ordem fixa (set não tem)
    out = HEADER.format(sources=body, packages=pkgs, digest=digest, app=strip_docstring(app_src))
    # garantia: o texto embutido é idêntico aos arquivos de origem
    tree = ast.parse(out)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_SOURCES":
            if ast.literal_eval(node.value) != sources:
                raise SystemExit("bundle não reproduz as fontes exatamente")
    return out


def main():
    out = build()
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != out:
            print("app_standalone.py está desatualizado: rode "
                  "`python tools/build_standalone.py` e faça commit.", file=sys.stderr)
            sys.exit(1)
        print("app_standalone.py em dia.")
        return
    OUT.write_text(out, encoding="utf-8")
    print(f"gerado {OUT.relative_to(ROOT)} ({len(out.splitlines())} linhas)")


if __name__ == "__main__":
    main()

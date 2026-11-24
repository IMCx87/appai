import streamlit as st

from appai import charts
from appai.bench import capacity, summarize_run
from appai.cost import CLOUD_PRESETS, cost_curve, fmt_cost, onprem_analysis
from appai.store import display_name
from appai.ui.chat import session_kpis
from appai.ui.state import get_store, slo

BASES = ["Vazão dentro do SLO", "Vazão de pico"]


def apply_cloud_preset():
    preset = CLOUD_PRESETS.get(st.session_state.price_preset)
    if preset:
        st.session_state.price_in_per_1m, st.session_state.price_out_per_1m = preset


def _sources() -> dict:
    out = {}
    cur = st.session_state.bench_result
    if cur:
        out["__current__"] = ("Benchmark atual", cur)
    for r in get_store().list("sweep", limit=50):
        if cur and r["id"] == cur.get("id"):
            continue
        out[r["id"]] = (display_name(r), None)
    return out


def _measured(run: dict, basis: str) -> dict:
    """Vazão e potência medidas de uma execução (pico ou limite do SLO)."""
    rows = summarize_run(run["data"]["levels"], slo())
    cap = capacity(rows)
    if basis == BASES[0]:
        load, tps = cap["max_ok"], cap["throughput"]
    else:
        best = max(rows, key=lambda r: r["throughput"] or 0)
        load, tps = best["load"], best["throughput"]
    per_level = (run["data"].get("infra") or {}).get("per_level") or []
    agg = next((a for a in per_level if a["load"] == load), {})
    power = next((v for k, v in agg.items()
                  if k.startswith("prom:") and ("power" in k.lower() or "(w)" in k.lower())
                  and k.endswith(":mean")), None)
    row = next((r for r in rows if r["load"] == load), None)
    kwh_m = (agg["energy_kwh"] / row["tokens_out"] * 1e6
             if agg.get("energy_kwh") and row and row["tokens_out"] else None)
    return {"tps": tps or 0, "load": load, "power_w": power, "kwh_per_m": kwh_m,
            "mode": run["info"].get("mode", "closed")}


def render():
    s = st.session_state
    st.caption(
        "Compara o custo por token em uma cloud pública com o custo de rodar a mesma carga "
        "on-prem. **Preços, custo de GPU, utilização e energia são premissas editáveis, não "
        "medições** — a vazão (e a potência, com DCGM) vem do benchmark."
    )
    st.markdown("##### ☁️ Cloud pública (US$ por 1M de tokens)")
    p1, p2, p3 = st.columns(3)
    p1.selectbox("Referência de preço", list(CLOUD_PRESETS), key="price_preset",
                 on_change=apply_cloud_preset,
                 help="Valores ilustrativos — não são a tabela de preços de nenhum provedor.")
    p2.number_input("Entrada (US$ / 1M tokens)", key="price_in_per_1m", min_value=0.0,
                    step=0.1, format="%.2f")
    p3.number_input("Saída (US$ / 1M tokens)", key="price_out_per_1m", min_value=0.0,
                    step=0.1, format="%.2f")

    k = session_kpis()
    if k["n"]:
        avg = k["cost"] / k["n"]
        q1, q2, q3 = st.columns(3)
        q1.metric("Custo médio por troca de mensagem", fmt_cost(avg))
        q2.metric("Custo desta sessão", fmt_cost(k["cost"]))
        q3.metric("Projeção: 1.000 trocas", f"US$ {avg * 1000:.2f}")

    st.markdown("##### 🖥️ On-prem (GPU própria) × Cloud")
    src = _sources()
    if not src:
        st.info("Rode o **Benchmark** para usar a vazão medida no cálculo on-prem.")
        return
    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Vazão medida de", list(src), format_func=lambda x: src[x][0])
    c2.radio("Base de capacidade", BASES, key="cost_basis",
             help="Dentro do SLO: a vazão no maior nível que ainda cumpre o SLO — o que "
                  "o endpoint entrega com qualidade. Pico: o máximo absoluto.")
    run = src[choice][1] or get_store().get(choice)
    meas = _measured(run, s.cost_basis)

    o1, o2, o3 = st.columns(3)
    o1.number_input("Custo por GPU-hora (US$)", key="gpu_hour", min_value=0.0, step=0.5,
                    format="%.2f", help="Amortização do hardware + operação (+ energia, se "
                                        "não for separada abaixo).")
    o2.number_input("Nº de GPUs do endpoint", key="gpu_count", min_value=1, step=1)
    o3.slider("Utilização média (%)", 5, 100, key="utilization_pct",
              help="Fração do mês em que a GPU trabalha na vazão medida. 100% é o teto "
                   "teórico; tráfego real tem picos e vales.")
    v1, v2 = st.columns(2)
    v1.number_input("Volume mensal — tokens de entrada (milhões)", key="monthly_in_m",
                    min_value=0.0, step=100.0)
    v2.number_input("Volume mensal — tokens de saída (milhões)", key="monthly_out_m",
                    min_value=0.0, step=50.0)
    with st.expander("⚡ Energia", expanded=s.energy_separate):
        st.toggle("Calcular energia à parte (o custo de GPU-hora não inclui energia)",
                  key="energy_separate")
        e1, e2, e3 = st.columns(3)
        e1.number_input("Potência média por GPU (W)", key="power_w", min_value=0.0, step=25.0)
        if meas["power_w"]:
            per_gpu = meas["power_w"] / max(int(s.gpu_count), 1)
            e1.caption(f"Medido (DCGM): {meas['power_w']:.0f} W no total ≈ {per_gpu:.0f} W/GPU")
        e2.number_input("PUE do datacenter", key="pue", min_value=1.0, step=0.05)
        e3.number_input("Preço da energia (US$/kWh)", key="kwh_price", min_value=0.0,
                        step=0.01, format="%.3f")
        if meas["kwh_per_m"]:
            st.caption(f"Energia medida no nível usado: **{meas['kwh_per_m']:.3f} kWh por 1M "
                       "tokens de saída** (só GPU, sem PUE).")

    if not meas["tps"]:
        st.warning("Nenhum nível cumpriu o SLO nesta execução: não há vazão com qualidade "
                   "para basear o custo. Use 'Vazão de pico' ou ajuste o SLO.")
        return
    params = {"peak_tps": meas["tps"], "utilization": s.utilization_pct / 100,
              "gpu_hour": s.gpu_hour, "gpu_count": s.gpu_count,
              "monthly_in_m": s.monthly_in_m, "monthly_out_m": s.monthly_out_m,
              "price_in": s.price_in_per_1m, "price_out": s.price_out_per_1m,
              "energy_separate": s.energy_separate, "power_w": s.power_w, "pue": s.pue,
              "kwh_price": s.kwh_price}
    a = onprem_analysis(params)
    unit = charts.load_unit(meas["mode"])

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Vazão usada", f"{meas['tps']:,.0f} tok/s",
              help=f"{s.cost_basis}: {meas['load']:g} {unit}")
    r1.caption(f"{s.cost_basis.lower()} · {meas['load']:g} {unit}")
    if a["onprem_per_m_out"] is not None:
        r2.metric("On-prem: US$ / 1M tokens de saída", f"{a['onprem_per_m_out']:.2f}",
                  delta=f"{a['onprem_per_m_out'] - a['cloud_per_m_out']:+.2f} vs cloud",
                  delta_color="inverse",
                  help="Com a entrada processada junto, na proporção do benchmark.")
    r3.metric("Cloud: US$ / 1M tokens de saída", f"{a['cloud_per_m_out']:.2f}",
              help="Preço de saída + entrada na proporção do volume mensal informado.")
    r4.metric("Capacidade de 1 endpoint", f"{a['capacity_out_m']:,.0f}M tok/mês",
              help="Vazão × 730 h × utilização.")

    m1, m2, m3 = st.columns(3)
    m1.metric("On-prem / mês", f"US$ {a['onprem_month']:,.0f}",
              help=f"{a['replicas']} endpoint(s) × US$ {a['set_month']:,.0f}"
                   + (f" (energia US$ {a['energy_month']:,.0f})" if a["energy_month"] else ""))
    m1.caption(f"{a['replicas']} endpoint(s) de {s.gpu_count} GPU(s)")
    m2.metric("Cloud / mês", f"US$ {a['cloud_month']:,.0f}")
    m3.metric("Diferença / mês", f"US$ {a['savings_month']:,.0f}",
              help="Positivo: on-prem mais barato para este volume.")

    msg = ""
    if a["breakeven_m"]:
        msg = f"Break-even: **{a['breakeven_m']:,.0f}M tokens de saída/mês**"
        if not a["breakeven_within_capacity"]:
            msg += (" — acima da capacidade de um endpoint nesta utilização: com estas "
                    "premissas a cloud fica mais barata em qualquer volume.")
        else:
            msg += "."
    st.markdown(msg)
    st.plotly_chart(charts.cost_curve_fig(cost_curve(params), s.monthly_out_m),
                    width="stretch", key="cost-curve")
    st.caption(
        "Simplificações: a vazão medida embute a proporção entrada/saída do benchmark; "
        "on-prem escala em endpoints inteiros (degraus); não inclui rede, storage, "
        "licenças nem pessoas — some ao custo de GPU-hora se quiser."
    )

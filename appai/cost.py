"""Modelo de custo: cloud pública (por token) × on-prem (GPU própria).

Tudo aqui são premissas editáveis, não medições — exceto a vazão (tok/s) e,
quando há telemetria DCGM, a potência medida durante o benchmark.
"""

import math

HOURS_PER_MONTH = 730

# Preços ILUSTRATIVOS por 1M tokens (entrada, saída) em US$ — ajuste ao provedor.
CLOUD_PRESETS = {
    "Modelo pequeno (~8B)": (0.15, 0.60),
    "Modelo médio (~70B)": (0.90, 0.90),
    "Modelo frontier": (3.00, 15.00),
    "Personalizado": None,
}


def estimate_cost(in_tokens: int, out_tokens: int, prices: tuple) -> float:
    """Custo estimado se a mesma troca rodasse em uma cloud pública
    (preços em US$ por 1M de tokens: entrada, saída)."""
    return (in_tokens or 0) / 1e6 * prices[0] + (out_tokens or 0) / 1e6 * prices[1]


def fmt_cost(cost: float) -> str:
    return f"US$ {cost:.5f}" if cost < 0.01 else f"US$ {cost:.4f}"


def energy_month_usd(power_w: float, gpu_count: int, pue: float, kwh_price: float) -> float:
    """Energia de um conjunto de GPUs por mês (potência média × PUE)."""
    return power_w * gpu_count * pue * HOURS_PER_MONTH / 1000 * kwh_price


def onprem_analysis(p: dict) -> dict:
    """Compara on-prem × cloud para um volume mensal.

    p: peak_tps (tok/s de saída medidos), utilization (0–1), gpu_hour, gpu_count,
       monthly_in_m, monthly_out_m (milhões de tokens), price_in, price_out,
       energy_separate (bool), power_w, pue, kwh_price.

    A vazão medida já embute a proporção entrada/saída do benchmark; a
    capacidade é expressa em tokens de saída, processando a entrada junto.
    """
    peak = max(float(p.get("peak_tps") or 0), 0.0)
    util = min(max(float(p.get("utilization", 1.0)), 0.01), 1.0)
    gpu_count = max(int(p.get("gpu_count", 1)), 1)
    vol_in = max(float(p.get("monthly_in_m", 0)), 0.0)
    vol_out = max(float(p.get("monthly_out_m", 0)), 0.0)
    price_in, price_out = float(p.get("price_in", 0)), float(p.get("price_out", 0))

    hw_month = float(p.get("gpu_hour", 0)) * gpu_count * HOURS_PER_MONTH
    energy_month = 0.0
    if p.get("energy_separate"):
        energy_month = energy_month_usd(float(p.get("power_w", 0)), gpu_count,
                                        float(p.get("pue", 1.0)), float(p.get("kwh_price", 0)))
    set_month = hw_month + energy_month  # custo mensal de UM endpoint (conjunto de GPUs)

    capacity_out_m = peak * 3600 * HOURS_PER_MONTH * util / 1e6
    replicas = math.ceil(vol_out / capacity_out_m) if capacity_out_m > 0 and vol_out > 0 else 1
    replicas = max(replicas, 1)
    onprem_month = set_month * replicas

    ratio = vol_in / vol_out if vol_out > 0 else 0.0
    cloud_per_m_out = price_out + ratio * price_in  # custo cloud por 1M de saída (com a entrada)
    cloud_month = vol_in * price_in + vol_out * price_out

    onprem_per_m_out = set_month / capacity_out_m if capacity_out_m > 0 else None
    breakeven_m = set_month / cloud_per_m_out if cloud_per_m_out > 0 else None

    return {
        "hw_month": hw_month,
        "energy_month": energy_month,
        "set_month": set_month,
        "capacity_out_m": capacity_out_m,
        "replicas": replicas,
        "onprem_month": onprem_month,
        "cloud_month": cloud_month,
        "cloud_per_m_out": cloud_per_m_out,
        "onprem_per_m_out": onprem_per_m_out,
        "breakeven_m": breakeven_m,
        "breakeven_within_capacity": (breakeven_m is not None
                                      and breakeven_m <= capacity_out_m),
        "in_out_ratio": ratio,
        "savings_month": cloud_month - onprem_month,
    }


def cost_curve(p: dict, points: int = 40) -> list:
    """Custo mensal cloud × on-prem para volumes de 0 a ~3× o informado (ou a
    capacidade), para o gráfico de break-even. On-prem é uma escada: cada vez
    que o volume passa da capacidade, entra mais um conjunto de GPUs."""
    base = onprem_analysis(p)
    cap = base["capacity_out_m"]
    top = max(float(p.get("monthly_out_m", 0)) * 3, cap * 2.5, 1.0)
    ratio = base["in_out_ratio"]
    out = []
    for i in range(points + 1):
        vol = top * i / points
        reps = max(math.ceil(vol / cap), 1) if cap > 0 else 1
        out.append({
            "volume_out_m": vol,
            "cloud": vol * float(p.get("price_out", 0)) + vol * ratio * float(p.get("price_in", 0)),
            "onprem": base["set_month"] * reps,
        })
    return out


def energy_per_m_tokens(energy_kwh: float, out_tokens: int) -> float | None:
    """kWh por 1M de tokens de saída, a partir da energia medida (DCGM)."""
    if not out_tokens or energy_kwh is None:
        return None
    return energy_kwh / out_tokens * 1e6

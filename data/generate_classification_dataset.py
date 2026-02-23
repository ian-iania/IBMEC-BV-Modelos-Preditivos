#!/usr/bin/env python3
"""Generate synthetic PF default 30+ classification dataset for FP&A classes.

How to run:
    python data/generate_classification_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_ROWS = 30000


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _build_selic(dt: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Create selic linked to time with trend, cycle, and mild noise."""
    month_idx = (dt.dt.year - dt.dt.year.min()) * 12 + (dt.dt.month - 1)
    month_idx = month_idx.to_numpy(dtype=float)

    trend = 7.8 + 0.028 * month_idx
    cycle = 1.45 * np.sin(2 * np.pi * month_idx / 18 + 0.8)
    noise = rng.normal(0.0, 0.35, size=len(dt))

    return np.clip(trend + cycle + noise, 5.0, 14.0)


def _calibrate_intercept(z_without_intercept: np.ndarray, target_min: float, target_max: float) -> tuple[float, float]:
    """Calibrate logistic intercept so sampled default rate falls in desired interval."""
    trial_intercepts = [-4.0, -3.8, -3.6, -3.4, -3.2, -3.0, -2.8]

    best_b0 = trial_intercepts[0]
    best_gap = 1e9
    best_rate = 0.0

    for b0 in trial_intercepts:
        mean_rate = float(_sigmoid(b0 + z_without_intercept).mean())
        if target_min <= mean_rate <= target_max:
            return b0, mean_rate

        center = 0.5 * (target_min + target_max)
        gap = abs(mean_rate - center)
        if gap < best_gap:
            best_gap = gap
            best_b0 = b0
            best_rate = mean_rate

    return best_b0, best_rate


def build_dataset(seed: int = SEED, n_rows: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start = np.datetime64("2019-01-01")
    end = np.datetime64("2024-12-31")
    days = (end - start).astype("timedelta64[D]").astype(int)

    dt = pd.to_datetime(start + rng.integers(0, days + 1, size=n_rows).astype("timedelta64[D]"))

    df = pd.DataFrame(
        {
            "id": np.arange(1, n_rows + 1, dtype=int),
            "dt": dt,
        }
    )

    # Core profile features.
    df["idade"] = rng.integers(18, 76, size=n_rows)
    renda = rng.lognormal(mean=np.log(5200), sigma=0.55, size=n_rows)
    df["renda_mensal"] = np.clip(renda, 1500.0, 30000.0)

    raw_score = rng.normal(loc=670, scale=110, size=n_rows)
    df["score_interno"] = np.clip(np.round(raw_score), 300, 950).astype(int)

    valor_solicitado = 2000 + rng.gamma(shape=2.1, scale=8500, size=n_rows)
    df["valor_solicitado"] = np.clip(valor_solicitado, 2000.0, 80000.0)

    df["prazo_meses"] = rng.choice([12, 24, 36, 48, 60], size=n_rows, p=[0.18, 0.27, 0.25, 0.18, 0.12])
    df["canal"] = rng.choice(["digital", "agencia", "parceiro"], size=n_rows, p=[0.45, 0.35, 0.20]).astype(str)

    util = rng.beta(a=2.5, b=2.5, size=n_rows)
    df["utilizacao_limite"] = np.clip(util, 0.0, 1.0)

    # Sort by date for temporal split and then build macro linked features.
    df = df.sort_values("dt", kind="mergesort").reset_index(drop=True)
    df["id"] = np.arange(1, n_rows + 1, dtype=int)

    df["selic"] = _build_selic(df["dt"], rng)

    # Delinquency drivers.
    score_norm = (df["score_interno"].to_numpy() - 650) / 120
    util_arr = df["utilizacao_limite"].to_numpy()

    p30 = _sigmoid(-2.05 - 0.75 * score_norm + 1.15 * (util_arr - 0.5))
    p30 = np.clip(p30, 0.03, 0.42)
    atraso_30 = rng.binomial(1, p30)

    p60 = _sigmoid(-3.20 - 0.95 * score_norm + 0.35 * atraso_30)
    p60 = np.clip(p60, 0.01, 0.22)
    atraso_60 = rng.binomial(1, p60)

    df["atraso_antes_30d"] = atraso_30.astype(int)
    df["atraso_antes_60d"] = atraso_60.astype(int)

    # Pricing linked to macro + risk.
    selic = df["selic"].to_numpy()
    taxa = (
        0.0105
        + 0.00125 * (selic - 8.0)
        + 0.0029 * ((700 - df["score_interno"].to_numpy()) / 100)
        + 0.0060 * df["atraso_antes_30d"].to_numpy()
        + 0.0090 * df["atraso_antes_60d"].to_numpy()
        + 0.0016 * (util_arr - 0.5)
        + rng.normal(0.0, 0.0016, size=n_rows)
    )
    df["taxa_mensal"] = np.clip(taxa, 0.008, 0.060)

    prazo = df["prazo_meses"].to_numpy()
    valor = df["valor_solicitado"].to_numpy()
    parcela = (valor / prazo) * (1.0 + df["taxa_mensal"].to_numpy() * (prazo / 12.0) * 0.6)
    df["parcela_mensal"] = parcela

    pti = parcela / df["renda_mensal"].to_numpy()
    df["pti"] = np.clip(pti, 0.05, 0.80)

    # Target score z with linear + nonlinear terms.
    pti_arr = df["pti"].to_numpy()
    prazo_scaled = (prazo - 24) / 12.0

    linear_part = (
        0.24 * (selic - 8.0)
        + 3.15 * (pti_arr - 0.25)
        - 0.62 * ((df["score_interno"].to_numpy() - 650) / 100)
        + 1.38 * df["atraso_antes_30d"].to_numpy()
        + 2.48 * df["atraso_antes_60d"].to_numpy()
        + 1.02 * (util_arr - 0.5)
        + 0.12 * prazo_scaled
    )

    pti_threshold_penalty = np.where(pti_arr > 0.35, 0.95 * (pti_arr - 0.35) * ((700 - df["score_interno"].to_numpy()) / 100), 0.0)
    selic_pti_interaction = 0.42 * np.maximum(selic - 10.0, 0.0) * np.maximum(pti_arr - 0.30, 0.0)
    score_delay60_interaction = 0.50 * ((df["score_interno"].to_numpy() - 650) / 100) * df["atraso_antes_60d"].to_numpy()

    z_wo_intercept = linear_part + pti_threshold_penalty + selic_pti_interaction + score_delay60_interaction

    intercept, _ = _calibrate_intercept(z_wo_intercept, target_min=0.08, target_max=0.12)

    p_default = _sigmoid(intercept + z_wo_intercept)
    p_default = np.clip(p_default, 0.001, 0.95)
    df["default_30p"] = rng.binomial(1, p_default).astype(int)

    # Financial columns for cost/reward analysis.
    ead = np.maximum(0.70 * valor, 1000.0)
    lgd = np.clip(rng.beta(a=5.2, b=4.2, size=n_rows), 0.25, 0.80)

    df["ead"] = ead
    df["lgd"] = lgd
    df["loss_if_default"] = df["ead"].to_numpy() * df["lgd"].to_numpy()

    tenor_factor = np.clip(df["prazo_meses"].to_numpy() / 24.0, 0.5, 2.5)
    profit_good = valor * df["taxa_mensal"].to_numpy() * 4.00 * tenor_factor
    df["profit_if_good"] = np.clip(profit_good, 150.0, 12000.0)

    # Final column order.
    ordered_cols = [
        "id",
        "dt",
        "selic",
        "idade",
        "renda_mensal",
        "score_interno",
        "valor_solicitado",
        "prazo_meses",
        "canal",
        "utilizacao_limite",
        "atraso_antes_30d",
        "atraso_antes_60d",
        "taxa_mensal",
        "parcela_mensal",
        "pti",
        "default_30p",
        "ead",
        "lgd",
        "loss_if_default",
        "profit_if_good",
    ]

    df = df[ordered_cols].sort_values("dt", kind="mergesort").reset_index(drop=True)
    df["id"] = np.arange(1, len(df) + 1, dtype=int)
    return df


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(seed=SEED, n_rows=N_ROWS)

    csv_path = data_dir / "bv_pf_default30_sintetico.csv"
    xlsx_path = data_dir / "bv_pf_default30_sintetico.xlsx"

    df_out = df.copy()
    df_out["dt"] = pd.to_datetime(df_out["dt"]).dt.strftime("%Y-%m-%d")

    df_out.to_csv(csv_path, index=False)
    df_out.to_excel(xlsx_path, index=False)

    print("Arquivos gerados:")
    print(f"- {csv_path}")
    print(f"- {xlsx_path}")

    print("\nResumo de validação:")
    print(f"- taxa média de default: {df['default_30p'].mean():.4f}")
    print(f"- selic min/max: {df['selic'].min():.4f} / {df['selic'].max():.4f}")
    print("- describe de pti, score_interno, loss_if_default:")
    print(df[["pti", "score_interno", "loss_if_default"]].describe())


if __name__ == "__main__":
    main()

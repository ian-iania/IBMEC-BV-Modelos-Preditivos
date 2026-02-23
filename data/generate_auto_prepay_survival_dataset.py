#!/usr/bin/env python3
"""Generate synthetic AUTO prepay survival dataset for FP&A classes.

How to run:
    python data/generate_auto_prepay_survival_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_ROWS = 30000
CUTOFF_DT = pd.Timestamp("2025-01-31")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _months_until_cutoff(orig_dt: pd.Series, cutoff_dt: pd.Timestamp) -> np.ndarray:
    """Return full months elapsed from origination date to cutoff date."""
    ydiff = cutoff_dt.year - orig_dt.dt.year.to_numpy()
    mdiff = cutoff_dt.month - orig_dt.dt.month.to_numpy()
    months = ydiff * 12 + mdiff

    before_day = cutoff_dt.day < orig_dt.dt.day.to_numpy()
    months = months - before_day.astype(int)
    return np.maximum(months, 0).astype(int)


def _build_selic_at_orig(orig_dt: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Create a plausible monthly Selic path linked to time and mild seasonality."""
    month_idx = (orig_dt.dt.year - 2019) * 12 + (orig_dt.dt.month - 1)
    month_idx = month_idx.to_numpy(dtype=float)

    trend = 7.1 + 0.055 * month_idx
    cycle = 1.4 * np.sin(2.0 * np.pi * month_idx / 20.0 + 0.4)
    noise = rng.normal(0.0, 0.30, size=len(orig_dt))

    return np.clip(trend + cycle + noise, 5.0, 14.0)


def build_dataset(seed: int = SEED, n_rows: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start = np.datetime64("2019-01-01")
    end = np.datetime64("2024-12-31")
    n_days = (end - start).astype("timedelta64[D]").astype(int)

    orig_dt = pd.to_datetime(start + rng.integers(0, n_days + 1, size=n_rows).astype("timedelta64[D]"))

    df = pd.DataFrame(
        {
            "id": np.arange(1, n_rows + 1, dtype=int),
            "orig_dt": orig_dt,
        }
    )

    df = df.sort_values("orig_dt", kind="mergesort").reset_index(drop=True)
    df["id"] = np.arange(1, len(df) + 1, dtype=int)

    df["term_meses"] = rng.choice([12, 24, 36, 48, 60], size=n_rows, p=[0.14, 0.23, 0.28, 0.21, 0.14])

    car_value = 18000.0 + rng.gamma(shape=3.4, scale=19000.0, size=n_rows)
    df["car_value"] = np.clip(car_value, 25000.0, 180000.0)

    ltv_base = rng.normal(loc=0.84, scale=0.14, size=n_rows)
    ltv_tail = rng.binomial(1, 0.11, size=n_rows) * rng.uniform(0.10, 0.34, size=n_rows)
    ltv = np.clip(ltv_base + ltv_tail, 0.50, 1.20)
    df["ltv"] = ltv

    df["loan_amount"] = np.clip(df["car_value"].to_numpy() * df["ltv"].to_numpy(), 12000.0, 210000.0)

    df["selic_at_orig"] = _build_selic_at_orig(df["orig_dt"], rng)

    df["idade"] = rng.integers(18, 76, size=n_rows)

    renda = rng.lognormal(mean=np.log(6200.0), sigma=0.56, size=n_rows)
    df["renda_mensal"] = np.clip(renda, 1500.0, 30000.0)

    score = rng.normal(loc=690, scale=115, size=n_rows)
    df["score_interno"] = np.clip(np.round(score), 300, 950).astype(int)

    df["canal"] = rng.choice(["digital", "agencia", "parceiro"], size=n_rows, p=[0.44, 0.34, 0.22]).astype(str)

    score_risk = (700.0 - df["score_interno"].to_numpy()) / 100.0
    ltv_risk = df["ltv"].to_numpy() - 0.82

    taxa = (
        0.0130
        + 0.00135 * (df["selic_at_orig"].to_numpy() - 8.0)
        + 0.0024 * score_risk
        + 0.0072 * ltv_risk
        + rng.normal(0.0, 0.0018, size=n_rows)
    )
    df["taxa_mensal_contrato"] = np.clip(taxa, 0.008, 0.055)

    term = df["term_meses"].to_numpy(dtype=float)
    loan = df["loan_amount"].to_numpy()
    taxa_m = df["taxa_mensal_contrato"].to_numpy()

    parcela_base = loan / term
    parcela_juros = loan * taxa_m * 0.85
    parcela_prazo = loan * (term / 60.0) * 0.0030
    parcela = parcela_base + parcela_juros + parcela_prazo

    df["parcela_mensal"] = np.clip(parcela, 250.0, 22000.0)

    pti = df["parcela_mensal"].to_numpy() / df["renda_mensal"].to_numpy()
    df["pti"] = np.clip(pti, 0.05, 0.80)

    # Survival core: score -> monthly hazard, then geometric time to event.
    ltv_arr = df["ltv"].to_numpy()
    selic_arr = df["selic_at_orig"].to_numpy()

    low_ltv_bonus = np.maximum(0.75 - ltv_arr, 0.0) * 2.6
    high_ltv_penalty = np.maximum(ltv_arr - 0.95, 0.0) * 3.2

    prepay_score = (
        -1.90
        + 0.33 * (9.0 - selic_arr)
        + 18.0 * (df["taxa_mensal_contrato"].to_numpy() - 0.020)
        + low_ltv_bonus
        - high_ltv_penalty
        + 0.16 * (df["canal"].to_numpy() == "digital").astype(float)
        + rng.normal(0.0, 0.25, size=n_rows)
    )

    p_mensal = np.clip(_sigmoid(prepay_score) * 0.19, 0.004, 0.20)

    tempo_evento = rng.geometric(p_mensal).astype(int)
    evento_no_prazo = tempo_evento <= df["term_meses"].to_numpy()

    meses_observaveis = _months_until_cutoff(df["orig_dt"], CUTOFF_DT)
    tempo_obs_max = np.minimum(df["term_meses"].to_numpy(), meses_observaveis)

    evento_observado = evento_no_prazo & (tempo_evento <= meses_observaveis)
    t_meses = np.where(evento_observado, tempo_evento, tempo_obs_max)

    df["T_meses"] = np.maximum(t_meses, 0).astype(int)
    df["E_prepay"] = evento_observado.astype(int)

    ordered_cols = [
        "id",
        "orig_dt",
        "term_meses",
        "car_value",
        "loan_amount",
        "ltv",
        "selic_at_orig",
        "idade",
        "renda_mensal",
        "score_interno",
        "canal",
        "taxa_mensal_contrato",
        "parcela_mensal",
        "pti",
        "T_meses",
        "E_prepay",
    ]

    return df[ordered_cols]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(seed=SEED, n_rows=N_ROWS)

    csv_path = data_dir / "bv_auto_prepay_survival_sintetico.csv"
    xlsx_path = data_dir / "bv_auto_prepay_survival_sintetico.xlsx"

    df_out = df.copy()
    df_out["orig_dt"] = pd.to_datetime(df_out["orig_dt"]).dt.strftime("%Y-%m-%d")

    df_out.to_csv(csv_path, index=False)
    df_out.to_excel(xlsx_path, index=False)

    print("Arquivos gerados:")
    print(f"- {csv_path}")
    print(f"- {xlsx_path}")

    print("\nResumo de validação:")
    print(f"- taxa de evento observado (E_prepay mean): {df['E_prepay'].mean():.4f}")
    print(f"- taxa de censura (E_prepay==0): {(df['E_prepay'] == 0).mean():.4f}")
    print(
        "- estatísticas T_meses (min/mediana/max): "
        f"{int(df['T_meses'].min())} / {float(df['T_meses'].median()):.1f} / {int(df['T_meses'].max())}"
    )
    print(
        "- ltv min/média/max: "
        f"{df['ltv'].min():.4f} / {df['ltv'].mean():.4f} / {df['ltv'].max():.4f}"
    )
    print(f"- selic_at_orig min/max: {df['selic_at_orig'].min():.4f} / {df['selic_at_orig'].max():.4f}")
    print("- head(3):")
    print(df.head(3))


if __name__ == "__main__":
    main()

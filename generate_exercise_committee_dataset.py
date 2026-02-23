#!/usr/bin/env python3
"""Gera dataset sintético para exercício de cutoff por custo no comitê de crédito."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_ROWS = 12000
PROFIT_SCALE = 8.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _build_selic(dt: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Cria SELIC dependente do tempo entre 2019 e 2024."""
    month_idx = (dt.dt.year - dt.dt.year.min()) * 12 + (dt.dt.month - 1)
    month_idx = month_idx.to_numpy(dtype=float)

    trend = 6.1 + 0.068 * month_idx
    cycle = 2.10 * np.sin(2 * np.pi * month_idx / 18 + 0.6)
    noise = rng.normal(0.0, 0.35, size=len(dt))

    return np.clip(trend + cycle + noise, 5.0, 13.0)


def _calibrate_intercept(z_without_intercept: np.ndarray, target_rate: float = 0.10) -> float:
    """Ajusta intercepto para atingir taxa média de default desejada."""
    low, high = -8.0, 3.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        mean_rate = float(_sigmoid(mid + z_without_intercept).mean())
        if mean_rate > target_rate:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def build_dataset(seed: int = SEED, n_rows: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start = np.datetime64("2019-01-01")
    end = np.datetime64("2024-12-31")
    days = (end - start).astype("timedelta64[D]").astype(int)

    dt = pd.to_datetime(start + rng.integers(0, days + 1, size=n_rows).astype("timedelta64[D]"))
    df = pd.DataFrame({"id": np.arange(1, n_rows + 1, dtype=int), "dt": dt})

    df = df.sort_values("dt", kind="mergesort").reset_index(drop=True)
    df["id"] = np.arange(1, len(df) + 1, dtype=int)

    df["selic"] = _build_selic(df["dt"], rng)

    df["idade"] = rng.integers(18, 76, size=n_rows)

    renda = rng.lognormal(mean=np.log(5600), sigma=0.58, size=n_rows)
    df["renda_mensal"] = np.clip(renda, 1500.0, 30000.0)

    raw_score = (
        rng.normal(loc=675, scale=90, size=n_rows)
        + 0.0024 * (df["renda_mensal"].to_numpy() - 5600.0)
        + 0.35 * (df["idade"].to_numpy() - 40.0)
        + rng.normal(0.0, 24.0, size=n_rows)
    )
    df["score_interno"] = np.clip(np.round(raw_score), 300, 950).astype(int)

    df["valor_solicitado"] = np.clip(
        df["renda_mensal"].to_numpy() * rng.uniform(0.9, 5.2, size=n_rows) + rng.gamma(shape=2.2, scale=2400, size=n_rows),
        2000.0,
        80000.0,
    )

    df["prazo_meses"] = rng.choice([12, 24, 36, 48, 60], size=n_rows, p=[0.21, 0.29, 0.24, 0.16, 0.10])
    df["canal"] = rng.choice(["digital", "agencia", "parceiro"], size=n_rows, p=[0.50, 0.33, 0.17]).astype(str)

    util = (
        rng.beta(a=2.1, b=2.8, size=n_rows)
        + 0.10 * ((680.0 - df["score_interno"].to_numpy()) / 350.0)
        + 0.04 * (df["canal"].to_numpy() == "parceiro")
    )
    df["utilizacao_limite"] = np.clip(util, 0.0, 1.0)

    score_arr = df["score_interno"].to_numpy()
    util_arr = df["utilizacao_limite"].to_numpy()

    p30 = _sigmoid(-2.55 + 1.10 * (util_arr - 0.50) + 0.78 * ((690.0 - score_arr) / 120.0) + 0.22 * (df["canal"].to_numpy() == "parceiro"))
    p30 = np.clip(p30, 0.01, 0.55)
    atraso_30 = rng.binomial(1, p30).astype(int)

    p60 = _sigmoid(-4.15 + 1.45 * atraso_30 + 0.62 * ((700.0 - score_arr) / 120.0) + 0.30 * (util_arr - 0.50))
    p60 = np.clip(p60, 0.003, 0.22)
    atraso_60 = rng.binomial(1, p60).astype(int)

    df["atraso_antes_30d"] = atraso_30
    df["atraso_antes_60d"] = atraso_60

    selic_arr = df["selic"].to_numpy()
    taxa = (
        0.0120
        + 0.00105 * (selic_arr - 7.5)
        + 0.00245 * ((700.0 - score_arr) / 100.0)
        + 0.0047 * df["atraso_antes_30d"].to_numpy()
        + 0.0082 * df["atraso_antes_60d"].to_numpy()
        + 0.0018 * (util_arr - 0.50)
        + 0.0009 * (df["canal"].to_numpy() == "parceiro")
        - 0.0003 * (df["canal"].to_numpy() == "digital")
        + rng.normal(0.0, 0.0013, size=n_rows)
    )
    df["taxa_mensal"] = np.clip(taxa, 0.008, 0.060)

    prazo_arr = df["prazo_meses"].to_numpy()
    valor_arr = df["valor_solicitado"].to_numpy()
    taxa_arr = df["taxa_mensal"].to_numpy()

    amort_denom = np.maximum(1.0 - np.power(1.0 + taxa_arr, -prazo_arr), 1e-6)
    parcela_raw = valor_arr * taxa_arr / amort_denom
    pti = np.clip(parcela_raw / df["renda_mensal"].to_numpy(), 0.05, 0.80)
    df["pti"] = pti
    df["parcela_mensal"] = df["pti"].to_numpy() * df["renda_mensal"].to_numpy()

    score_norm = (score_arr - 650.0) / 120.0
    prazo_scaled = (prazo_arr - 24.0) / 12.0

    linear_part = (
        0.20 * (selic_arr - 8.0)
        + 3.05 * (pti - 0.25)
        - 0.72 * score_norm
        + 1.28 * df["atraso_antes_30d"].to_numpy()
        + 2.55 * df["atraso_antes_60d"].to_numpy()
        + 0.88 * (util_arr - 0.50)
        + 0.11 * prazo_scaled
        + 0.18 * (df["canal"].to_numpy() == "parceiro")
    )

    pti_threshold_penalty = 1.15 * np.maximum(pti - 0.35, 0.0)
    selic_pti_interaction = 0.55 * np.maximum(selic_arr - 10.0, 0.0) * np.maximum(pti - 0.30, 0.0)
    score_loses_power_when_60d = 0.60 * score_norm * df["atraso_antes_60d"].to_numpy()

    z_without_intercept = linear_part + pti_threshold_penalty + selic_pti_interaction + score_loses_power_when_60d
    intercept = _calibrate_intercept(z_without_intercept, target_rate=0.10)

    p_default = _sigmoid(intercept + z_without_intercept)
    p_default = np.clip(p_default, 0.001, 0.95)
    df["default_30p"] = rng.binomial(1, p_default).astype(int)

    df["ead"] = np.maximum(0.70 * df["valor_solicitado"].to_numpy(), 1000.0)
    df["lgd"] = np.clip(rng.beta(a=3.2, b=3.4, size=n_rows), 0.25, 0.80)
    df["loss_if_default"] = df["ead"].to_numpy() * df["lgd"].to_numpy()

    tenor_factor = np.clip(df["prazo_meses"].to_numpy() / 24.0, 0.5, 2.5)
    profit_if_good = (
        df["valor_solicitado"].to_numpy()
        * df["taxa_mensal"].to_numpy()
        * 0.60
        * tenor_factor
        * PROFIT_SCALE
    )
    df["profit_if_good"] = np.clip(profit_if_good, 150.0, 12000.0)

    ordered_cols = [
        "id",
        "dt",
        "selic",
        "idade",
        "renda_mensal",
        "score_interno",
        "valor_solicitado",
        "prazo_meses",
        "taxa_mensal",
        "parcela_mensal",
        "pti",
        "utilizacao_limite",
        "atraso_antes_30d",
        "atraso_antes_60d",
        "canal",
        "default_30p",
        "ead",
        "lgd",
        "loss_if_default",
        "profit_if_good",
    ]

    return df[ordered_cols].sort_values("dt", kind="mergesort").reset_index(drop=True)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(seed=SEED, n_rows=N_ROWS)

    csv_path = data_dir / "ex1_pf_committee.csv"
    xlsx_path = data_dir / "ex1_pf_committee.xlsx"

    df_out = df.copy()
    df_out["dt"] = pd.to_datetime(df_out["dt"]).dt.strftime("%Y-%m-%d")
    df_out.to_csv(csv_path, index=False)
    df_out.to_excel(xlsx_path, index=False)

    mean_default = df["default_30p"].mean()
    mean_loss = df["loss_if_default"].mean()
    mean_profit = df["profit_if_good"].mean()
    ratio = mean_loss / mean_profit

    print("Arquivos gerados:")
    print(f"- {csv_path}")
    print(f"- {xlsx_path}")

    print("\nValidações:")
    print(f"- taxa média de default: {mean_default:.4f}")
    print(f"- média loss_if_default: {mean_loss:.2f}")
    print(f"- média profit_if_good: {mean_profit:.2f}")
    print(f"- razão loss/profit: {ratio:.2f}")
    print(f"- selic min/max: {df['selic'].min():.2f} / {df['selic'].max():.2f}")
    print("- head(3):")
    print(df.head(3))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate synthetic FP&A datasets for regression and time-series classes.

This script creates three datasets in both CSV and XLSX formats under `data/`:
- bv_fpa_regressao_linear
- bv_fpa_regressao_nonlinear
- bv_fpa_timeseries

How to run:
    python scripts/generate_datasets_regression.py

The datasets are monthly (start of month), 72 rows (2019-01-01 to 2024-12-01),
and are designed for educational use in Colab notebooks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _base_calendar(periods: int = 72) -> pd.DataFrame:
    """Return the base monthly calendar and shared drivers."""
    ds = pd.date_range("2019-01-01", periods=periods, freq="MS")
    t = np.arange(periods)

    # Shared macro and business drivers with plausible dynamics.
    selic = 6.2 + 0.055 * t + 1.6 * np.sin(2 * np.pi * (t + 2) / 18)
    desemprego = 10.1 - 0.012 * t + 0.7 * np.sin(2 * np.pi * (t + 5) / 24)
    ltv_medio = 74.5 + 0.05 * t + 2.3 * np.sin(2 * np.pi * (t + 1) / 16)
    marketing = 62 + 9.5 * np.sin(2 * np.pi * (t + 1) / 12) + 3.3 * np.sin(2 * np.pi * t / 6)
    mix_auto = 0.44 + 0.0018 * t + 0.035 * np.sin(2 * np.pi * t / 12)

    # spread correlated with selic by construction.
    spread = 2.05 + 0.34 * selic + 0.12 * np.sin(2 * np.pi * t / 12)

    df = pd.DataFrame(
        {
            "ds": ds,
            "selic": selic,
            "desemprego": desemprego,
            "ltv_medio": ltv_medio,
            "spread": spread,
            "marketing": marketing,
            "mix_auto": mix_auto,
            "mes": ds.month,
        }
    )

    # Clamp to reasonable teaching ranges.
    df["selic"] = df["selic"].clip(4.5, 16.5)
    df["desemprego"] = df["desemprego"].clip(7.0, 13.5)
    df["ltv_medio"] = df["ltv_medio"].clip(68.0, 84.0)
    df["spread"] = df["spread"].clip(2.5, 8.5)
    df["marketing"] = df["marketing"].clip(38.0, 92.0)
    df["mix_auto"] = df["mix_auto"].clip(0.30, 0.75)

    return df


def build_regressao_linear(seed: int = 42) -> pd.DataFrame:
    """Create near-linear FP&A regression dataset with mild seasonality and noise."""
    rng = np.random.default_rng(seed)
    df = _base_calendar().copy()

    t = np.arange(len(df))
    sazonal = 7.0 * np.sin(2 * np.pi * t / 12)
    ruido = rng.normal(0, 3.5, size=len(df))

    y = (
        155
        + 0.72 * df["marketing"].to_numpy()
        + 34.0 * df["mix_auto"].to_numpy()
        - 4.4 * df["selic"].to_numpy()
        - 2.6 * df["desemprego"].to_numpy()
        - 0.9 * df["spread"].to_numpy()
        - 0.55 * (df["ltv_medio"].to_numpy() - 75)
        + sazonal
        + ruido
    )

    df["y"] = np.maximum(y, 10)

    out_cols = ["ds", "y", "selic", "desemprego", "ltv_medio", "spread", "marketing", "mix_auto", "mes"]
    return df[out_cols]


def build_regressao_nonlinear(seed: int = 43) -> pd.DataFrame:
    """Create nonlinear FP&A dataset with threshold, interaction, and extra noise."""
    rng = np.random.default_rng(seed)
    df = _base_calendar().copy()

    t = np.arange(len(df))
    sazonal = 7.8 * np.sin(2 * np.pi * (t + 1) / 12)

    base = (
        158
        + 0.67 * df["marketing"].to_numpy()
        + 30.0 * df["mix_auto"].to_numpy()
        - 4.1 * df["selic"].to_numpy()
        - 2.5 * df["desemprego"].to_numpy()
        - 1.0 * df["spread"].to_numpy()
        - 0.45 * (df["ltv_medio"].to_numpy() - 75)
        + sazonal
    )

    selic = df["selic"].to_numpy()
    ltv = df["ltv_medio"].to_numpy()
    mix = df["mix_auto"].to_numpy()

    # Smooth nonlinearity: stronger penalty as selic rises above 11.
    selic_excesso = np.maximum(selic - 11.0, 0.0)
    penal_selic = 1.7 * (selic_excesso**1.6)

    # Threshold penalty when both conditions are high.
    limiar = (selic > 12.0) & (ltv > 78.0)
    penal_limiar = np.where(limiar, 11.0 + 1.15 * (selic - 12.0) + 0.75 * (ltv - 78.0), 0.0)

    # Interaction: positive mix_auto effect gets weaker when selic is high.
    penal_interacao = np.maximum(selic - 10.5, 0.0) * (mix - 0.40) * 14.0

    ruido_extra = rng.normal(0, 5.1, size=len(df))

    y = base - penal_selic - penal_limiar - penal_interacao + ruido_extra
    df["y"] = np.maximum(y, 10)

    out_cols = ["ds", "y", "selic", "desemprego", "ltv_medio", "spread", "marketing", "mix_auto", "mes"]
    return df[out_cols]


def build_timeseries(seed: int = 42) -> pd.DataFrame:
    """Create monthly univariate time series with exogenous signals for ARIMAX/Prophet."""
    rng = np.random.default_rng(seed)

    periods = 72
    ds = pd.date_range("2019-01-01", periods=periods, freq="MS")
    t = np.arange(periods)

    selic = 6.0 + 0.06 * t + 1.2 * np.sin(2 * np.pi * (t + 2) / 20)
    selic = np.clip(selic, 4.5, 16.0)

    # Campaign months: March, June, November + one specific shock month.
    evento = np.isin(ds.month, [3, 6, 11]).astype(int)
    evento[ds == pd.Timestamp("2021-04-01")] = 1

    tendencia = 172 + 1.05 * t
    sazonal = 10.0 * np.sin(2 * np.pi * t / 12) + 3.5 * np.cos(2 * np.pi * t / 12)
    efeito_evento = 8.0 * evento
    choque_unico = np.where(ds == pd.Timestamp("2021-04-01"), -22.0, 0.0)
    efeito_selic = -2.25 * (selic - selic.mean())
    ruido = rng.normal(0, 3.4, size=periods)

    y = tendencia + sazonal + efeito_evento + choque_unico + efeito_selic + ruido
    y = np.maximum(y, 10)

    return pd.DataFrame({"ds": ds, "y": y, "selic": selic, "evento": evento})


def _save_pair(df: pd.DataFrame, base_name: str, data_dir: Path) -> tuple[Path, Path]:
    """Save dataframe as CSV and XLSX and return output paths."""
    csv_path = data_dir / f"{base_name}.csv"
    xlsx_path = data_dir / f"{base_name}.xlsx"

    df_out = df.copy()
    df_out["ds"] = pd.to_datetime(df_out["ds"]).dt.strftime("%Y-%m-%d")

    df_out.to_csv(csv_path, index=False)
    df_out.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path


def main() -> None:
    """Generate and save all regression/time-series datasets."""
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    linear_df = build_regressao_linear(seed=42)
    nonlinear_df = build_regressao_nonlinear(seed=43)
    ts_df = build_timeseries(seed=42)

    outputs = []
    outputs.extend(_save_pair(linear_df, "bv_fpa_regressao_linear", data_dir))
    outputs.extend(_save_pair(nonlinear_df, "bv_fpa_regressao_nonlinear", data_dir))
    outputs.extend(_save_pair(ts_df, "bv_fpa_timeseries", data_dir))

    print("Arquivos gerados:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()

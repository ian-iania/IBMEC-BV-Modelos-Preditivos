#!/usr/bin/env python3
"""Generate synthetic AUTO clustering dataset for K-Means + PCA demo.

How to run:
    python data/generate_nb04_clustering_auto_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_ROWS = 12000


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _generate_dt_monthly(rng: np.random.Generator, n_rows: int) -> pd.Series:
    months = pd.date_range("2022-01-01", "2024-12-01", freq="MS")
    dt = rng.choice(months.to_numpy(), size=n_rows, replace=True)
    return pd.to_datetime(dt)


def _sample_cluster_sizes(n_rows: int) -> dict[str, int]:
    # Keep one dominant "massa" group while preserving clear separation.
    proportions = {"A": 0.26, "B": 0.36, "C": 0.18, "D": 0.20}
    sizes = {k: int(v * n_rows) for k, v in proportions.items()}
    remainder = n_rows - sum(sizes.values())
    sizes["B"] += remainder
    return sizes


def _build_cluster_a(rng: np.random.Generator, n: int) -> pd.DataFrame:
    idade = np.clip(rng.normal(29, 5, size=n), 18, 42).round().astype(int)
    renda = np.clip(rng.lognormal(np.log(5600), 0.33, size=n), 2200.0, 18000.0)
    car_value = np.clip(rng.lognormal(np.log(52000), 0.32, size=n), 25000.0, 105000.0)
    ltv = np.clip(rng.normal(0.78, 0.04, size=n), 0.70, 0.86)
    term = rng.choice([24, 36, 48], size=n, p=[0.60, 0.35, 0.05]).astype(int)
    score = np.clip(rng.normal(790, 55, size=n), 640, 950).round().astype(int)
    util = np.clip(rng.beta(2.1, 5.8, size=n), 0.0, 1.0)
    atraso30 = rng.binomial(1, 0.03, size=n).astype(int)
    atraso60 = rng.binomial(1, 0.004, size=n).astype(int)
    canal = rng.choice(["digital", "agencia", "parceiro"], size=n, p=[0.82, 0.10, 0.08]).astype(str)

    return pd.DataFrame(
        {
            "cluster_true": "A",
            "idade": idade,
            "renda_mensal": renda,
            "car_value": car_value,
            "ltv": ltv,
            "term_meses": term,
            "score_interno": score,
            "utilizacao_limite": util,
            "atraso_antes_30d": atraso30,
            "atraso_antes_60d": atraso60,
            "canal": canal,
        }
    )


def _build_cluster_b(rng: np.random.Generator, n: int) -> pd.DataFrame:
    idade = np.clip(rng.normal(39, 8, size=n), 22, 62).round().astype(int)
    renda = np.clip(rng.lognormal(np.log(7000), 0.36, size=n), 2500.0, 22000.0)
    car_value = np.clip(rng.lognormal(np.log(70000), 0.33, size=n), 30000.0, 140000.0)
    ltv = np.clip(rng.normal(0.89, 0.05, size=n), 0.80, 0.97)
    term = rng.choice([24, 36, 48, 60], size=n, p=[0.10, 0.64, 0.20, 0.06]).astype(int)
    score = np.clip(rng.normal(675, 70, size=n), 500, 860).round().astype(int)
    util = np.clip(rng.beta(2.7, 3.0, size=n), 0.0, 1.0)
    atraso30 = rng.binomial(1, 0.09, size=n).astype(int)
    atraso60 = rng.binomial(1, 0.018, size=n).astype(int)
    canal = rng.choice(["digital", "agencia", "parceiro"], size=n, p=[0.22, 0.36, 0.42]).astype(str)

    return pd.DataFrame(
        {
            "cluster_true": "B",
            "idade": idade,
            "renda_mensal": renda,
            "car_value": car_value,
            "ltv": ltv,
            "term_meses": term,
            "score_interno": score,
            "utilizacao_limite": util,
            "atraso_antes_30d": atraso30,
            "atraso_antes_60d": atraso60,
            "canal": canal,
        }
    )


def _build_cluster_c(rng: np.random.Generator, n: int) -> pd.DataFrame:
    idade = np.clip(rng.normal(43, 7, size=n), 26, 68).round().astype(int)
    renda = np.clip(rng.lognormal(np.log(14500), 0.30, size=n), 7000.0, 30000.0)
    car_value = np.clip(rng.lognormal(np.log(118000), 0.27, size=n), 70000.0, 180000.0)
    ltv = np.clip(rng.normal(1.04, 0.06, size=n), 0.95, 1.16)
    term = rng.choice([36, 48, 60], size=n, p=[0.20, 0.65, 0.15]).astype(int)
    score = np.clip(rng.normal(735, 50, size=n), 620, 920).round().astype(int)
    util = np.clip(rng.beta(3.1, 2.5, size=n), 0.0, 1.0)
    atraso30 = rng.binomial(1, 0.05, size=n).astype(int)
    atraso60 = rng.binomial(1, 0.010, size=n).astype(int)
    canal = rng.choice(["digital", "agencia", "parceiro"], size=n, p=[0.18, 0.34, 0.48]).astype(str)

    return pd.DataFrame(
        {
            "cluster_true": "C",
            "idade": idade,
            "renda_mensal": renda,
            "car_value": car_value,
            "ltv": ltv,
            "term_meses": term,
            "score_interno": score,
            "utilizacao_limite": util,
            "atraso_antes_30d": atraso30,
            "atraso_antes_60d": atraso60,
            "canal": canal,
        }
    )


def _build_cluster_d(rng: np.random.Generator, n: int) -> pd.DataFrame:
    idade = np.clip(rng.normal(37, 9, size=n), 20, 72).round().astype(int)
    renda = np.clip(rng.lognormal(np.log(4700), 0.42, size=n), 1500.0, 17000.0)
    car_value = np.clip(rng.lognormal(np.log(62000), 0.36, size=n), 25000.0, 125000.0)
    ltv = np.clip(rng.normal(0.99, 0.09, size=n), 0.82, 1.20)
    term = rng.choice([36, 48, 60], size=n, p=[0.10, 0.20, 0.70]).astype(int)
    score = np.clip(rng.normal(560, 75, size=n), 300, 780).round().astype(int)
    util = np.clip(rng.beta(5.2, 1.8, size=n), 0.0, 1.0)
    atraso30 = rng.binomial(1, 0.35, size=n).astype(int)
    atraso60 = rng.binomial(1, 0.13, size=n).astype(int)
    canal = rng.choice(["digital", "agencia", "parceiro"], size=n, p=[0.24, 0.52, 0.24]).astype(str)

    return pd.DataFrame(
        {
            "cluster_true": "D",
            "idade": idade,
            "renda_mensal": renda,
            "car_value": car_value,
            "ltv": ltv,
            "term_meses": term,
            "score_interno": score,
            "utilizacao_limite": util,
            "atraso_antes_30d": atraso30,
            "atraso_antes_60d": atraso60,
            "canal": canal,
        }
    )


def _build_pd_proxy(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    cluster_shift = df["cluster_true"].map({"A": -2.05, "B": -1.15, "C": -0.85, "D": 0.70}).to_numpy()

    z = (
        -2.35
        + 3.1 * (df["pti"].to_numpy() - 0.25)
        + 2.1 * (df["ltv"].to_numpy() - 0.85)
        + 0.9 * (df["taxa_mensal"].to_numpy() - 0.020) / 0.010
        + 1.0 * df["atraso_antes_30d"].to_numpy()
        + 1.5 * df["atraso_antes_60d"].to_numpy()
        + 0.95 * (650 - df["score_interno"].to_numpy()) / 200.0
        + cluster_shift
        + rng.normal(0.0, 0.18, size=len(df))
    )

    pd_proxy = 0.15 * _sigmoid(z)
    return np.clip(pd_proxy, 0.0, 0.15)


def build_dataset(seed: int = SEED, n_rows: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    sizes = _sample_cluster_sizes(n_rows)
    parts = [
        _build_cluster_a(rng, sizes["A"]),
        _build_cluster_b(rng, sizes["B"]),
        _build_cluster_c(rng, sizes["C"]),
        _build_cluster_d(rng, sizes["D"]),
    ]
    df = pd.concat(parts, ignore_index=True)

    df["dt"] = _generate_dt_monthly(rng, len(df))

    loan_noise = rng.normal(1.0, 0.035, size=len(df))
    loan_amount = df["car_value"].to_numpy() * df["ltv"].to_numpy() * loan_noise
    df["loan_amount"] = np.clip(loan_amount, 12000.0, 210000.0)

    taxa = (
        0.010
        + 0.0024 * (700 - df["score_interno"].to_numpy()) / 100.0
        + 0.0060 * (df["ltv"].to_numpy() - 0.80)
        + 0.0070 * df["atraso_antes_30d"].to_numpy()
        + 0.0105 * df["atraso_antes_60d"].to_numpy()
        + 0.0018 * (df["utilizacao_limite"].to_numpy() - 0.45)
        + df["cluster_true"].map({"A": -0.0012, "B": 0.0000, "C": 0.0012, "D": 0.0040}).to_numpy()
        + rng.normal(0.0, 0.0015, size=len(df))
    )
    df["taxa_mensal"] = np.clip(taxa, 0.008, 0.055)

    term = df["term_meses"].to_numpy(dtype=float)
    loan = df["loan_amount"].to_numpy()
    parcela = (loan / term) + (loan * df["taxa_mensal"].to_numpy() * 0.80)
    df["parcela_mensal"] = np.clip(parcela, 250.0, 23000.0)

    pti_raw = df["parcela_mensal"].to_numpy() / df["renda_mensal"].to_numpy()
    pti = np.clip(pti_raw, 0.05, 0.80)

    # Preserve the high-risk latent profile with systematically higher PTI.
    d_mask = df["cluster_true"].to_numpy() == "D"
    pti[d_mask] = np.clip(pti[d_mask] + rng.uniform(0.09, 0.18, size=d_mask.sum()), 0.12, 0.80)
    df["pti"] = pti

    df["pd_proxy"] = _build_pd_proxy(df, rng)

    exposure_flag = (df["loan_amount"].to_numpy() >= 120000.0) & (df["ltv"].to_numpy() >= 1.00)
    risk_band = np.where(
        exposure_flag,
        "exposicao",
        np.where(df["pd_proxy"].to_numpy() < 0.02, "baixo", np.where(df["pd_proxy"].to_numpy() <= 0.06, "medio", "alto")),
    )
    df["risk_band"] = risk_band.astype(str)

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["id"] = np.arange(1, len(df) + 1, dtype=int)

    ordered_cols = [
        "id",
        "dt",
        "car_value",
        "loan_amount",
        "ltv",
        "term_meses",
        "taxa_mensal",
        "parcela_mensal",
        "pti",
        "idade",
        "renda_mensal",
        "score_interno",
        "utilizacao_limite",
        "atraso_antes_30d",
        "atraso_antes_60d",
        "canal",
        "pd_proxy",
        "risk_band",
        "cluster_true",
    ]
    return df[ordered_cols]


def _cluster_profile_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("cluster_true", as_index=True)
        .agg(
            pd_proxy_mean=("pd_proxy", "mean"),
            loan_amount_mean=("loan_amount", "mean"),
            term_meses_mean=("term_meses", "mean"),
            ltv_mean=("ltv", "mean"),
        )
        .round(4)
    )

    canal_top1 = df.groupby("cluster_true")["canal"].agg(lambda x: x.value_counts().idxmax()).rename("canal_dominante")
    return grouped.join(canal_top1)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(seed=SEED, n_rows=N_ROWS)

    csv_path = data_dir / "bv_auto_clustering_sintetico.csv"
    xlsx_path = data_dir / "bv_auto_clustering_sintetico.xlsx"

    df_out = df.copy()
    df_out["dt"] = pd.to_datetime(df_out["dt"]).dt.strftime("%Y-%m-%d")

    df_out.to_csv(csv_path, index=False)
    df_out.to_excel(xlsx_path, index=False)

    print("Distribuicao de cluster_true (counts e %):")
    counts = df["cluster_true"].value_counts().sort_index()
    perc = (counts / len(df) * 100).round(2)
    dist = pd.DataFrame({"count": counts, "pct": perc})
    print(dist)

    print("\nMedias por cluster_true:")
    print(_cluster_profile_summary(df))

    print("\nRange de pd_proxy (min/max):")
    print(f"{df['pd_proxy'].min():.4f} / {df['pd_proxy'].max():.4f}")

    print("\nArquivos gerados:")
    print(f"- {csv_path}")
    print(f"- {xlsx_path}")


if __name__ == "__main__":
    main()

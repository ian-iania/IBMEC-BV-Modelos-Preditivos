#!/usr/bin/env python3
"""Generate synthetic monthly contract dataset for Markov bucket migration demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_CONTRATOS = 8000
N_MESES_HIST = 18
BUCKETS = ["OK", "Aten", "Mau", "WriteOff"]
LGD_MAP = {
    "OK": 0.00,
    "Aten": 0.05,
    "Mau": 0.25,
    "WriteOff": 0.80,
}


def _base_transition_matrix() -> dict[str, dict[str, float]]:
    return {
        "OK": {"OK": 0.92, "Aten": 0.06, "Mau": 0.02, "WriteOff": 0.00},
        "Aten": {"OK": 0.45, "Aten": 0.40, "Mau": 0.13, "WriteOff": 0.02},
        "Mau": {"OK": 0.08, "Aten": 0.15, "Mau": 0.60, "WriteOff": 0.17},
        "WriteOff": {"OK": 0.00, "Aten": 0.00, "Mau": 0.00, "WriteOff": 1.00},
    }


def _stress_transition_matrix(base: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    stress = {row: probs.copy() for row, probs in base.items()}

    stress["OK"]["Aten"] += 0.02
    stress["OK"]["OK"] -= 0.02

    stress["Aten"]["Mau"] += 0.02
    stress["Aten"]["Aten"] -= 0.02

    stress["Mau"]["WriteOff"] += 0.03
    stress["Mau"]["Mau"] -= 0.03

    for row in BUCKETS:
        row_sum = sum(stress[row][col] for col in BUCKETS)
        for col in BUCKETS:
            stress[row][col] = stress[row][col] / row_sum

    stress["WriteOff"] = {"OK": 0.00, "Aten": 0.00, "Mau": 0.00, "WriteOff": 1.00}
    return stress


def _scenario_by_month(months: pd.DatetimeIndex) -> dict[pd.Timestamp, str]:
    scenario_map = {m: "base" for m in months}
    last_six = list(months[-6:])
    stress_months = {last_six[1], last_six[3], last_six[5]}
    for m in stress_months:
        scenario_map[m] = "stress"
    return scenario_map


def _draw_next_bucket(
    rng: np.random.Generator,
    current_bucket: str,
    matrix: dict[str, dict[str, float]],
) -> str:
    probs = np.array([matrix[current_bucket][b] for b in BUCKETS], dtype=float)
    return str(rng.choice(BUCKETS, p=probs))


def build_dataset(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range(start="2023-01-01", periods=N_MESES_HIST, freq="MS")
    scenario_map = _scenario_by_month(months)

    base_matrix = _base_transition_matrix()
    stress_matrix = _stress_transition_matrix(base_matrix)

    initial_bucket_probs = np.array([0.90, 0.08, 0.02, 0.00], dtype=float)
    ead_0 = np.clip(rng.lognormal(mean=np.log(35000.0), sigma=0.55, size=N_CONTRATOS), 5000.0, 120000.0)

    rows: list[dict[str, object]] = []

    for contract_id in range(1, N_CONTRATOS + 1):
        bucket = str(rng.choice(BUCKETS, p=initial_bucket_probs))
        balance = float(ead_0[contract_id - 1])
        in_writeoff = bucket == "WriteOff"

        for t, month in enumerate(months):
            if t > 0:
                month_scenario = scenario_map[month]
                matrix = stress_matrix if month_scenario == "stress" else base_matrix
                bucket = _draw_next_bucket(rng, bucket, matrix)

            if not in_writeoff:
                amort_factor = max(1.0 - 0.012 * t, 0.68)
                noise = rng.normal(loc=1.0, scale=0.012)
                balance = float(np.clip(ead_0[contract_id - 1] * amort_factor * noise, 1000.0, None))

            if bucket == "WriteOff" and not in_writeoff:
                in_writeoff = True
                balance = max(balance, 500.0)

            rows.append(
                {
                    "contract_id": contract_id,
                    "month": month,
                    "ead": round(balance, 2),
                    "lgd_bucket": LGD_MAP[bucket],
                    "bucket": bucket,
                    "scenario": scenario_map[month],
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["contract_id", "month"], kind="mergesort").reset_index(drop=True)
    return df


def _estimated_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.sort_values(["contract_id", "month"], kind="mergesort").copy()
    tmp["bucket_next"] = tmp.groupby("contract_id")["bucket"].shift(-1)
    trans = tmp.dropna(subset=["bucket_next"]).copy()
    trans["bucket_next"] = trans["bucket_next"].astype(str)

    ctab = pd.crosstab(trans["bucket"], trans["bucket_next"])
    ctab = ctab.reindex(index=BUCKETS, columns=BUCKETS, fill_value=0)
    return ctab.div(ctab.sum(axis=1), axis=0).fillna(0.0)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(seed=SEED)

    csv_path = data_dir / "bv_markov_buckets_contratos_mensal.csv"
    xlsx_path = data_dir / "bv_markov_buckets_contratos_mensal.xlsx"

    df_out = df.copy()
    df_out["month"] = pd.to_datetime(df_out["month"]).dt.strftime("%Y-%m-%d")

    df_out.to_csv(csv_path, index=False)
    df_out.to_excel(xlsx_path, index=False)

    last_month = df["month"].max()
    dist_last = (
        df.loc[df["month"] == last_month, "bucket"]
        .value_counts(normalize=True)
        .reindex(BUCKETS, fill_value=0.0)
        .mul(100)
        .round(2)
    )
    p_hat = _estimated_transition_matrix(df).round(4)

    print("Validações:")
    print(f"1) linhas: {len(df):,} | contratos únicos: {df['contract_id'].nunique():,}")
    print("\n2) distribuição de buckets no último mês (%):")
    print(dist_last.to_string())
    print("\n3) matriz de transição estimada (bucket_t -> bucket_t+1):")
    print(p_hat.to_string())
    print("\n4) arquivos salvos:")
    print(f"- {csv_path}")
    print(f"- {xlsx_path}")


if __name__ == "__main__":
    main()

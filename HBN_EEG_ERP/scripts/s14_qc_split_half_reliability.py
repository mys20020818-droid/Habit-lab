from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from s00_utils_common import ensure_parent


def spearman_brown(r: float) -> float:
    if np.isnan(r) or r <= -1:
        return np.nan
    return (2 * r) / (1 + r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    rows: list[dict] = []
    for (task, feature), df in features.groupby(["task", "feature"]):
        df = df.dropna(subset=["split_odd_uv", "split_even_uv"])
        if len(df) < 5:
            r = np.nan
        else:
            r = df["split_odd_uv"].corr(df["split_even_uv"])
        rows.append(
            {
                "task": task,
                "feature": feature,
                "n_subject_task": len(df),
                "pearson_r_odd_even": r,
                "spearman_brown": spearman_brown(r),
                "mean_n_epochs": float(df["n_epochs"].mean()) if len(df) else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values(["task", "feature"])
    ensure_parent(args.out)
    out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()


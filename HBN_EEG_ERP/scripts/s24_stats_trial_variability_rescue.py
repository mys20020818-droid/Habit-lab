from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from s00_utils_common import ensure_parent


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7


FEATURES = ["p1", "n1", "p3"]
TASKS = ["CCD", "SuS"]
COMPONENTS = ["iiv_general", "iiv_specific_sus_minus_ccd"]


def pooled_subject_task_iiv(recording_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    df = recording_features.copy()
    for col in ["n_epochs", "amplitude_uv", "amplitude_uv_sd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["task"].isin(TASKS)].dropna(subset=["n_epochs", "amplitude_uv", "amplitude_uv_sd"])
    df = df[df["n_epochs"] >= 2].copy()

    for (subject, task, feature, channel_group), sub in df.groupby(["subject", "task", "feature", "channel_group"]):
        ns = sub["n_epochs"].to_numpy(dtype=float)
        means = sub["amplitude_uv"].to_numpy(dtype=float)
        sds = sub["amplitude_uv_sd"].to_numpy(dtype=float)
        total_n = float(ns.sum())
        if total_n <= 1:
            continue
        pooled_mean = float(np.sum(ns * means) / total_n)
        ss_within = np.sum((ns - 1) * (sds**2))
        ss_between = np.sum(ns * (means - pooled_mean) ** 2)
        pooled_var = float((ss_within + ss_between) / (total_n - 1))
        if pooled_var <= 0 or not np.isfinite(pooled_var):
            continue
        pooled_sd = float(np.sqrt(pooled_var))
        rows.append(
            {
                "subject": subject,
                "task": task,
                "feature": feature,
                "channel_group": channel_group,
                "n_recordings": int(sub["epochs_path"].nunique()),
                "n_epochs": int(total_n),
                "iiv_sd_uv": pooled_sd,
                "log_iiv_sd": float(np.log(pooled_sd)),
                "mean_amplitude_uv": pooled_mean,
            }
        )
    return pd.DataFrame(rows)


def zscore_by_task(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{value_col}_z"] = out.groupby("task")[value_col].transform(
        lambda s: (s - s.mean()) / s.std(ddof=1)
    )
    return out


def build_iiv_components(iiv: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for feature in FEATURES:
        sub = iiv[iiv["feature"].eq(feature)].copy()
        sub = zscore_by_task(sub, "log_iiv_sd")
        wide = sub.pivot(index="subject", columns="task", values="log_iiv_sd_z").dropna(subset=TASKS)
        nwide = sub.pivot(index="subject", columns="task", values="n_epochs").reindex(wide.index)
        for subject, row in wide.iterrows():
            rows.append(
                {
                    "subject": subject,
                    "feature": feature,
                    "iiv_general": float(np.mean([row["CCD"], row["SuS"]])),
                    "iiv_specific_sus_minus_ccd": float((row["SuS"] - row["CCD"]) / np.sqrt(2)),
                    "ccd_log_iiv_z": float(row["CCD"]),
                    "sus_log_iiv_z": float(row["SuS"]),
                    "ccd_n_epochs": int(nwide.loc[subject, "CCD"]),
                    "sus_n_epochs": int(nwide.loc[subject, "SuS"]),
                }
            )
    components = pd.DataFrame(rows)
    meta_cols = [
        "subject",
        "age",
        "age_group",
        "sex",
        "full_pheno",
        "p_factor",
        "attention",
        "internalizing",
        "externalizing",
    ]
    keep = [col for col in meta_cols if col in metadata.columns]
    components = components.merge(metadata[keep].drop_duplicates("subject"), on="subject", how="left")
    for col in ["age", "p_factor", "attention", "internalizing", "externalizing"]:
        if col in components.columns:
            components[col] = pd.to_numeric(components[col], errors="coerce")
    return components


def fit_iiv_models(components: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_rows: list[dict] = []
    interaction_rows: list[dict] = []
    for feature, fdf in components.groupby("feature"):
        df = fdf.copy()
        df["age_c"] = df["age"] - df["age"].mean()
        df["age_c2"] = df["age_c"] ** 2
        df["p_c"] = df["p_factor"] - df["p_factor"].mean()
        for outcome in COMPONENTS:
            model_df = df.dropna(subset=[outcome, "age_c", "age_c2", "sex", "p_c"])
            if len(model_df) < 30:
                continue
            base = smf.ols(f"{outcome} ~ age_c + age_c2 + C(sex)", data=model_df).fit()
            main = smf.ols(f"{outcome} ~ age_c + age_c2 + C(sex) + p_c", data=model_df).fit()
            inter = smf.ols(
                f"{outcome} ~ age_c + age_c2 + C(sex) + p_c + age_c:p_c + age_c2:p_c",
                data=model_df,
            ).fit()
            main_rows.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "term": "p_factor_main",
                    "estimate": float(main.params.get("p_c", np.nan)),
                    "std_error": float(main.bse.get("p_c", np.nan)),
                    "t_value": float(main.tvalues.get("p_c", np.nan)),
                    "p_value": float(main.pvalues.get("p_c", np.nan)),
                    "n": int(main.nobs),
                    "base_r2": float(base.rsquared),
                    "model_r2": float(main.rsquared),
                    "delta_r2": float(main.rsquared - base.rsquared),
                }
            )
            f_value, p_value, df_diff = inter.compare_f_test(main)
            interaction_rows.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "test": "age_and_age2_x_pfactor_after_main",
                    "df_diff": float(df_diff),
                    "f_value": float(f_value),
                    "p_value": float(p_value),
                    "n": int(inter.nobs),
                    "main_r2": float(main.rsquared),
                    "interaction_r2": float(inter.rsquared),
                    "delta_r2": float(inter.rsquared - main.rsquared),
                    "age_x_p_estimate": float(inter.params.get("age_c:p_c", np.nan)),
                    "age2_x_p_estimate": float(inter.params.get("age_c2:p_c", np.nan)),
                }
            )
    main_models = pd.DataFrame(main_rows)
    interactions = pd.DataFrame(interaction_rows)
    if not main_models.empty:
        main_models["q_value"] = multipletests(main_models["p_value"].fillna(1.0), method="fdr_bh")[1]
    if not interactions.empty:
        interactions["q_value"] = multipletests(interactions["p_value"].fillna(1.0), method="fdr_bh")[1]
    return main_models, interactions


def fit_task_iiv_models(iiv: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    df = iiv.merge(metadata.drop_duplicates("subject"), on="subject", how="left")
    for col in ["age", "p_factor", "log_iiv_sd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    rows: list[dict] = []
    for (task, feature), sub in df.groupby(["task", "feature"]):
        model_df = sub.dropna(subset=["log_iiv_sd", "age", "sex", "p_factor"]).copy()
        if len(model_df) < 30:
            continue
        model_df["age_c"] = model_df["age"] - model_df["age"].mean()
        model_df["age_c2"] = model_df["age_c"] ** 2
        model_df["p_c"] = model_df["p_factor"] - model_df["p_factor"].mean()
        model = smf.ols("log_iiv_sd ~ age_c + age_c2 + C(sex) + p_c", data=model_df).fit()
        rows.append(
            {
                "task": task,
                "feature": feature,
                "outcome": "log_iiv_sd",
                "estimate": float(model.params.get("p_c", np.nan)),
                "std_error": float(model.bse.get("p_c", np.nan)),
                "t_value": float(model.tvalues.get("p_c", np.nan)),
                "p_value": float(model.pvalues.get("p_c", np.nan)),
                "r2": float(model.rsquared),
                "n": int(model.nobs),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = multipletests(out["p_value"].fillna(1.0), method="fdr_bh")[1]
    return out


def plot_iiv_results(components: pd.DataFrame, main_models: pd.DataFrame, out_base: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6), sharex=True)
    for r, outcome in enumerate(COMPONENTS):
        all_y = pd.to_numeric(components[outcome], errors="coerce").dropna()
        y_lo, y_hi = np.percentile(all_y, [1, 99])
        pad = (y_hi - y_lo) * 0.15 if y_hi > y_lo else 0.2
        for c, feature in enumerate(FEATURES):
            ax = axes[r, c]
            sub = components[components["feature"].eq(feature)].dropna(subset=[outcome, "p_factor"])
            ax.scatter(sub["p_factor"], sub[outcome], s=8, alpha=0.24, color="#3775BA", linewidths=0)
            if len(sub) >= 5:
                coef = np.polyfit(sub["p_factor"], sub[outcome], 1)
                xs = np.linspace(sub["p_factor"].min(), sub["p_factor"].max(), 100)
                ax.plot(xs, coef[0] * xs + coef[1], color="#B64342", linewidth=1.2)
            hit = main_models[(main_models["feature"].eq(feature)) & (main_models["outcome"].eq(outcome))]
            if not hit.empty:
                row = hit.iloc[0]
                ax.text(
                    0.02,
                    0.96,
                    f"q={row['q_value']:.3g}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.2,
                )
            ax.axhline(0, color="#D8D8D8", linewidth=0.7)
            ax.set_ylim(y_lo - pad, y_hi + pad)
            ax.set_title(f"{feature.upper()} {'general' if r == 0 else 'specific'} IIV")
            if r == 1:
                ax.set_xlabel("p-factor")
            if c == 0:
                ax.set_ylabel("IIV component score")
    fig.suptitle("Trial-to-trial ERP variability components versus p-factor", y=1.01, fontsize=9, fontweight="bold")
    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    ensure_parent(str(out_base))
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording-features", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    args = parser.parse_args()

    recording_features = pd.read_csv(args.recording_features)
    metadata = pd.read_csv(args.metadata)

    iiv = pooled_subject_task_iiv(recording_features)
    components = build_iiv_components(iiv, metadata)
    main_models, interactions = fit_iiv_models(components)
    task_models = fit_task_iiv_models(iiv, metadata)

    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    iiv.to_csv(out_dir / "iiv_subject_task_features.csv", index=False)
    components.to_csv(out_dir / "iiv_components.csv", index=False)
    main_models.to_csv(out_dir / "iiv_pfactor_main_models.csv", index=False)
    interactions.to_csv(out_dir / "iiv_pfactor_age_interactions.csv", index=False)
    task_models.to_csv(out_dir / "iiv_task_pfactor_models.csv", index=False)
    plot_iiv_results(components, main_models, figures_dir / "iiv_pfactor_components")

    summary = pd.DataFrame(
        [
            {
                "table": "iiv_subject_task_features",
                "rows": len(iiv),
                "subjects": iiv["subject"].nunique() if not iiv.empty else 0,
            },
            {
                "table": "iiv_components",
                "rows": len(components),
                "subjects": components["subject"].nunique() if not components.empty else 0,
            },
        ]
    )
    summary.to_csv(out_dir / "iiv_sample_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(main_models.sort_values("p_value").to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from statsmodels.stats.multitest import multipletests

from s00_utils_common import ensure_parent
from s20_model_component_decomposition import (
    FEATURE_ORDER,
    PSYCH_VARS,
    age_group_anova,
    age_group_summary,
    build_components,
    component_reliability,
    fit_models,
    variance_decomposition,
)


AGE_GROUPS = ["child_5_9", "adolescent_10_14", "youth_15_21"]
COMPONENTS = ["component_general", "component_specific_sus_minus_ccd"]


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_num = (len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)
    pooled_den = len(a) + len(b) - 2
    pooled = np.sqrt(pooled_num / pooled_den) if pooled_den > 0 else np.nan
    return float((b.mean() - a.mean()) / pooled) if pooled and np.isfinite(pooled) else np.nan


def average_design_row(model, age_group: str, sex_weights: pd.Series) -> np.ndarray:
    design_info = model.model.data.design_info
    row = np.zeros(len(model.params), dtype=float)
    for sex, weight in sex_weights.items():
        new = pd.DataFrame({"age_group": [age_group], "sex": [sex]})
        x = np.asarray(build_design_matrices([design_info], new)[0])[0]
        row += float(weight) * x
    return row


def age_group_posthoc(components: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    valid = components[components["age_group"].isin(AGE_GROUPS)].copy()
    valid["age_group"] = pd.Categorical(valid["age_group"], categories=AGE_GROUPS, ordered=True)

    for feature in FEATURE_ORDER:
        feature_df = valid[valid["feature"].eq(feature)].copy()
        for component in COMPONENTS:
            model_df = feature_df.dropna(subset=[component, "age_group", "sex"]).copy()
            if len(model_df) < 20:
                continue
            model = smf.ols(f"{component} ~ C(age_group) + C(sex)", data=model_df).fit()
            sex_weights = model_df["sex"].value_counts(normalize=True)

            for group_a, group_b in combinations(AGE_GROUPS, 2):
                x_a = average_design_row(model, group_a, sex_weights)
                x_b = average_design_row(model, group_b, sex_weights)
                contrast = x_b - x_a
                test = model.t_test(contrast)
                a_values = model_df.loc[model_df["age_group"].eq(group_a), component]
                b_values = model_df.loc[model_df["age_group"].eq(group_b), component]
                rows.append(
                    {
                        "feature": feature,
                        "component": component,
                        "contrast": f"{group_b} - {group_a}",
                        "group_a": group_a,
                        "group_b": group_b,
                        "n_a": int(a_values.notna().sum()),
                        "n_b": int(b_values.notna().sum()),
                        "mean_a": float(a_values.mean()),
                        "mean_b": float(b_values.mean()),
                        "adjusted_difference": float(np.ravel(test.effect)[0]),
                        "std_error": float(np.ravel(test.sd)[0]),
                        "t_value": float(np.ravel(test.tvalue)[0]),
                        "p_value": float(test.pvalue),
                        "cohen_d_raw": cohen_d(a_values, b_values),
                        "model_r_squared": float(model.rsquared),
                        "n_model": int(model.nobs),
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = multipletests(out["p_value"].fillna(1.0), method="fdr_bh")[1]
    return out


def subject_task_qc(qc: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["n_visual_events", "n_epochs_kept"]
    for col in numeric_cols:
        qc[col] = pd.to_numeric(qc[col], errors="coerce").fillna(0)
    grouped = (
        qc.groupby(["subject", "task"], dropna=False)
        .agg(
            n_recordings=("raw_path", "count"),
            n_ok=("status", lambda s: int((s == "ok").sum())),
            n_failed=("status", lambda s: int((s != "ok").sum())),
            n_visual_events=("n_visual_events", "sum"),
            n_epochs_kept=("n_epochs_kept", "sum"),
        )
        .reset_index()
    )
    grouped["drop_fraction"] = np.where(
        grouped["n_visual_events"] > 0,
        1.0 - grouped["n_epochs_kept"] / grouped["n_visual_events"],
        np.nan,
    )
    return grouped


def filter_features_by_quality(
    features: pd.DataFrame,
    qc_subject_task: pd.DataFrame,
    min_epochs_per_task: int,
    max_drop_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    quality = qc_subject_task.copy()
    quality["passes_quality"] = (
        (quality["n_epochs_kept"] >= min_epochs_per_task)
        & (quality["drop_fraction"] <= max_drop_fraction)
    )
    keep_pairs = quality.loc[quality["passes_quality"], ["subject", "task"]].drop_duplicates()
    filtered = features.merge(keep_pairs.assign(_keep=1), on=["subject", "task"], how="left")
    filtered = filtered[filtered["_keep"].eq(1)].drop(columns=["_keep"])
    return filtered, quality


def write_subset_outputs(prefix: str, components: pd.DataFrame, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    age_models, psych_models = fit_models(components)
    outputs = {
        "components": components,
        "variance": variance_decomposition(features),
        "reliability": component_reliability(components),
        "age_models": age_models,
        "psych_models": psych_models,
        "age_group_summary": age_group_summary(components),
        "age_group_anova": age_group_anova(components),
    }
    for name, table in outputs.items():
        path = f"{prefix}_{name}.csv"
        ensure_parent(path)
        table.to_csv(path, index=False)
    return outputs


def core_replication_table(source_tables: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict] = []
    for label, table in source_tables:
        if table.empty:
            continue
        for _, row in table.iterrows():
            if row["component"] != "component_general":
                continue
            if row["feature"] not in ["n1", "p3"]:
                continue
            rows.append(
                {
                    "analysis": label,
                    "feature": row["feature"],
                    "component": row["component"],
                    "n": int(row["n"]),
                    "f_value": float(row["f_value"]),
                    "p_value": float(row["p_value"]),
                    "q_value": float(row["q_value"]),
                    "r_squared": float(row["r_squared"]),
                }
            )
    return pd.DataFrame(rows)


def make_age_group_plot(components: pd.DataFrame, out_path: str) -> None:
    plot_df = components[
        components["age_group"].isin(AGE_GROUPS)
        & components["feature"].isin(["n1", "p3"])
    ].copy()
    plot_df["age_group"] = pd.Categorical(plot_df["age_group"], categories=AGE_GROUPS, ordered=True)
    rng = np.random.default_rng(20260511)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, feature in zip(axes, ["n1", "p3"]):
        sub = plot_df[plot_df["feature"].eq(feature)]
        data = [
            sub.loc[sub["age_group"].eq(group), "component_general"].dropna().to_numpy()
            for group in AGE_GROUPS
        ]
        ax.boxplot(data, positions=np.arange(len(AGE_GROUPS)), widths=0.45, showfliers=False)
        for idx, values in enumerate(data):
            if len(values) == 0:
                continue
            jitter = rng.normal(0, 0.045, size=len(values))
            ax.scatter(np.full(len(values), idx) + jitter, values, s=9, alpha=0.25, color="#2f5d7c")
            mean = float(np.mean(values))
            sem = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
            ax.errorbar(idx, mean, yerr=1.96 * sem, fmt="o", color="#b8322c", capsize=4)
        ax.set_title(f"{feature.upper()} general component")
        ax.set_xticks(np.arange(len(AGE_GROUPS)))
        ax.set_xticklabels(["5-9", "10-14", "15-21"])
        ax.set_xlabel("Age group")
        ax.set_ylabel("Component score (task-z)")
        ax.axhline(0, color="0.75", linewidth=0.8)
    fig.tight_layout()
    ensure_parent(out_path)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--components", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--qc", required=True)
    parser.add_argument("--main-age-group-anova", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    parser.add_argument("--min-epochs-per-task", type=int, default=40)
    parser.add_argument("--max-drop-fraction", type=float, default=0.25)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features)
    components = pd.read_csv(args.components)
    metadata = pd.read_csv(args.metadata)
    qc = pd.read_csv(args.qc)
    main_anova = pd.read_csv(args.main_age_group_anova)

    posthoc = age_group_posthoc(components)
    posthoc.to_csv(out_dir / "full_sus_ccd_age_group_posthoc.csv", index=False)

    qc_subject_task = subject_task_qc(qc)
    high_features, quality = filter_features_by_quality(
        features,
        qc_subject_task,
        min_epochs_per_task=args.min_epochs_per_task,
        max_drop_fraction=args.max_drop_fraction,
    )
    quality.to_csv(out_dir / "full_sus_ccd_subject_task_quality.csv", index=False)

    high_components = build_components(high_features, metadata)
    high_outputs = write_subset_outputs(
        str(out_dir / "full_sus_ccd_high_quality"),
        high_components,
        high_features,
    )

    full_pheno_components = components[components["full_pheno"].eq("Yes")].copy()
    full_pheno_subjects = set(full_pheno_components["subject"].dropna().unique())
    full_pheno_features = features[features["subject"].isin(full_pheno_subjects)].copy()
    pheno_outputs = write_subset_outputs(
        str(out_dir / "full_sus_ccd_full_pheno"),
        full_pheno_components,
        full_pheno_features,
    )

    high_full_pheno_components = high_components[
        high_components["subject"].isin(full_pheno_subjects)
    ].copy()
    high_full_pheno_subjects = set(high_full_pheno_components["subject"].dropna().unique())
    high_full_pheno_features = high_features[
        high_features["subject"].isin(high_full_pheno_subjects)
    ].copy()
    high_pheno_outputs = write_subset_outputs(
        str(out_dir / "full_sus_ccd_high_quality_full_pheno"),
        high_full_pheno_components,
        high_full_pheno_features,
    )

    replication = core_replication_table(
        [
            ("main", main_anova),
            ("high_quality", high_outputs["age_group_anova"]),
            ("full_pheno", pheno_outputs["age_group_anova"]),
            ("high_quality_full_pheno", high_pheno_outputs["age_group_anova"]),
        ]
    )
    replication.to_csv(out_dir / "full_sus_ccd_core_age_replication.csv", index=False)

    make_age_group_plot(
        components,
        str(figures_dir / "full_sus_ccd_age_group_general_n1_p3.png"),
    )

    summary = pd.DataFrame(
        [
            {
                "analysis": "main_components",
                "subjects": components["subject"].nunique(),
                "component_rows": len(components),
                "feature_rows": len(features),
                "min_epochs_per_task": np.nan,
                "max_drop_fraction": np.nan,
            },
            {
                "analysis": "high_quality",
                "subjects": high_components["subject"].nunique(),
                "component_rows": len(high_components),
                "feature_rows": len(high_features),
                "min_epochs_per_task": args.min_epochs_per_task,
                "max_drop_fraction": args.max_drop_fraction,
            },
            {
                "analysis": "full_pheno",
                "subjects": full_pheno_components["subject"].nunique(),
                "component_rows": len(full_pheno_components),
                "feature_rows": len(full_pheno_features),
                "min_epochs_per_task": np.nan,
                "max_drop_fraction": np.nan,
            },
            {
                "analysis": "high_quality_full_pheno",
                "subjects": high_full_pheno_components["subject"].nunique(),
                "component_rows": len(high_full_pheno_components),
                "feature_rows": len(high_full_pheno_features),
                "min_epochs_per_task": args.min_epochs_per_task,
                "max_drop_fraction": args.max_drop_fraction,
            },
        ]
    )
    summary.to_csv(out_dir / "full_sus_ccd_followup_sample_summary.csv", index=False)


if __name__ == "__main__":
    main()

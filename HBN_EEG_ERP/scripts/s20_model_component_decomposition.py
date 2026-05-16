from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

from s00_utils_common import ensure_parent


PRIMARY_TASKS = ("CCD", "SuS")
FEATURE_ORDER = ["p1", "n1", "p3"]
PSYCH_VARS = ["p_factor", "attention", "internalizing", "externalizing"]


def zscore_by_task(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{value_col}_z"] = out.groupby("task")[value_col].transform(
        lambda s: (s - s.mean()) / s.std(ddof=1)
    )
    return out


def spearman_brown(r: float) -> float:
    if pd.isna(r) or r <= -1:
        return np.nan
    return (2 * r) / (1 + r)


def wide_for_feature(df: pd.DataFrame, feature: str, value_col: str) -> pd.DataFrame:
    sub = df[df["feature"].eq(feature)].copy()
    sub = zscore_by_task(sub, value_col)
    wide = sub.pivot(index="subject", columns="task", values=f"{value_col}_z")
    return wide.dropna(subset=list(PRIMARY_TASKS))


def build_components(features: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for feature in FEATURE_ORDER:
        amp = wide_for_feature(features, feature, "amplitude_uv")
        odd = wide_for_feature(features, feature, "split_odd_uv")
        even = wide_for_feature(features, feature, "split_even_uv")

        common_subjects = amp.index.intersection(odd.index).intersection(even.index)
        for subject in common_subjects:
            ccd = amp.loc[subject, "CCD"]
            sus = amp.loc[subject, "SuS"]
            odd_ccd = odd.loc[subject, "CCD"]
            odd_sus = odd.loc[subject, "SuS"]
            even_ccd = even.loc[subject, "CCD"]
            even_sus = even.loc[subject, "SuS"]
            rows.append(
                {
                    "subject": subject,
                    "feature": feature,
                    "component_general": float(np.mean([ccd, sus])),
                    "component_specific_sus_minus_ccd": float((sus - ccd) / np.sqrt(2)),
                    "general_odd": float(np.mean([odd_ccd, odd_sus])),
                    "general_even": float(np.mean([even_ccd, even_sus])),
                    "specific_odd": float((odd_sus - odd_ccd) / np.sqrt(2)),
                    "specific_even": float((even_sus - even_ccd) / np.sqrt(2)),
                    "ccd_z": float(ccd),
                    "sus_z": float(sus),
                }
            )
    components = pd.DataFrame(rows)
    keep = [
        col
        for col in [
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
        if col in metadata.columns
    ]
    components = components.merge(metadata[keep].drop_duplicates("subject"), on="subject", how="left")
    for col in ["age", *PSYCH_VARS]:
        if col in components.columns:
            components[col] = pd.to_numeric(components[col], errors="coerce")
    return components


def variance_decomposition(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for feature in FEATURE_ORDER:
        sub = features[(features["feature"].eq(feature)) & (features["task"].isin(PRIMARY_TASKS))].copy()
        sub = zscore_by_task(sub, "amplitude_uv")
        wide = sub.pivot(index="subject", columns="task", values="amplitude_uv_z").dropna(
            subset=list(PRIMARY_TASKS)
        )
        stacked = wide[list(PRIMARY_TASKS)].stack()
        subject_mean = wide[list(PRIMARY_TASKS)].mean(axis=1)
        general_repeated = pd.Series(
            np.repeat(subject_mean.values, len(PRIMARY_TASKS)),
            index=stacked.index,
        )
        residual = stacked - general_repeated
        total_var = float(stacked.var(ddof=1))
        general_var = float(general_repeated.var(ddof=1))
        specific_var = float(residual.var(ddof=1))
        rows.append(
            {
                "feature": feature,
                "n_subjects": len(wide),
                "total_variance_task_z": total_var,
                "general_variance": general_var,
                "specific_variance": specific_var,
                "icc_general": general_var / total_var if total_var else np.nan,
                "icc_specific": specific_var / total_var if total_var else np.nan,
            }
        )
    return pd.DataFrame(rows)


def component_reliability(components: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for feature, df in components.groupby("feature"):
        for component, odd_col, even_col in [
            ("general", "general_odd", "general_even"),
            ("specific_sus_minus_ccd", "specific_odd", "specific_even"),
        ]:
            valid = df.dropna(subset=[odd_col, even_col])
            r = valid[odd_col].corr(valid[even_col]) if len(valid) >= 5 else np.nan
            rows.append(
                {
                    "feature": feature,
                    "component": component,
                    "n_subjects": len(valid),
                    "pearson_r_odd_even": r,
                    "spearman_brown": spearman_brown(r),
                }
            )
    return pd.DataFrame(rows)


def fit_models(components: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    age_rows: list[dict] = []
    psych_rows: list[dict] = []
    for feature, df in components.groupby("feature"):
        df = df.copy()
        df["age2"] = df["age"] ** 2
        df = df.dropna(subset=["age", "sex"])
        for component in ["component_general", "component_specific_sus_minus_ccd"]:
            model_df = df.dropna(subset=[component])
            if len(model_df) < 20:
                continue
            age_model = smf.ols(f"{component} ~ age + age2 + C(sex)", data=model_df).fit()
            for term in ["age", "age2"]:
                age_rows.append(
                    {
                        "feature": feature,
                        "component": component,
                        "term": term,
                        "estimate": age_model.params.get(term, np.nan),
                        "std_error": age_model.bse.get(term, np.nan),
                        "p_value": age_model.pvalues.get(term, np.nan),
                        "r_squared": age_model.rsquared,
                        "n": int(age_model.nobs),
                    }
                )

            for psych in PSYCH_VARS:
                if psych not in model_df.columns:
                    continue
                psych_df = model_df.dropna(subset=[psych])
                if len(psych_df) < 20:
                    continue
                assoc = smf.ols(f"{component} ~ age + age2 + C(sex) + {psych}", data=psych_df).fit()
                psych_rows.append(
                    {
                        "feature": feature,
                        "component": component,
                        "predictor": psych,
                        "estimate": assoc.params.get(psych, np.nan),
                        "std_error": assoc.bse.get(psych, np.nan),
                        "p_value": assoc.pvalues.get(psych, np.nan),
                        "r_squared": assoc.rsquared,
                        "n": int(assoc.nobs),
                    }
                )
    age = pd.DataFrame(age_rows)
    psych = pd.DataFrame(psych_rows)
    if not psych.empty:
        psych["q_value"] = multipletests(psych["p_value"].fillna(1.0), method="fdr_bh")[1]
    if not age.empty:
        age["q_value"] = multipletests(age["p_value"].fillna(1.0), method="fdr_bh")[1]
    return age, psych


def age_group_summary(components: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (feature, group), df in components.groupby(["feature", "age_group"], dropna=False):
        for component in ["component_general", "component_specific_sus_minus_ccd"]:
            rows.append(
                {
                    "feature": feature,
                    "age_group": group,
                    "component": component,
                    "n": df[component].notna().sum(),
                    "mean": df[component].mean(),
                    "sd": df[component].std(ddof=1),
                    "sem": df[component].std(ddof=1) / np.sqrt(df[component].notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def age_group_anova(components: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for feature, df in components.groupby("feature"):
        df = df.dropna(subset=["age_group", "sex"])
        df = df[df["age_group"].isin(["child_5_9", "adolescent_10_14", "youth_15_21"])]
        for component in ["component_general", "component_specific_sus_minus_ccd"]:
            model_df = df.dropna(subset=[component])
            if len(model_df) < 20:
                continue
            model = smf.ols(f"{component} ~ C(age_group) + C(sex)", data=model_df).fit()
            table = anova_lm(model, typ=2)
            term = "C(age_group)"
            rows.append(
                {
                    "feature": feature,
                    "component": component,
                    "term": term,
                    "df": table.loc[term, "df"],
                    "f_value": table.loc[term, "F"],
                    "p_value": table.loc[term, "PR(>F)"],
                    "r_squared": model.rsquared,
                    "n": int(model.nobs),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = multipletests(out["p_value"].fillna(1.0), method="fdr_bh")[1]
    return out


def make_plots(components: pd.DataFrame, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for feature in FEATURE_ORDER:
        df = components[components["feature"].eq(feature)].dropna(subset=["age"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        for ax, component, title in [
            (axes[0], "component_general", "Task-general"),
            (axes[1], "component_specific_sus_minus_ccd", "Task-specific SuS-CCD"),
        ]:
            ax.scatter(df["age"], df[component], s=24, alpha=0.65)
            if len(df) >= 10:
                slope, intercept, r, p, _ = stats.linregress(df["age"], df[component])
                xs = np.linspace(df["age"].min(), df["age"].max(), 100)
                ax.plot(xs, intercept + slope * xs, color="black", linewidth=1.5)
                ax.set_title(f"{title}\nr={r:.2f}, p={p:.3g}")
            else:
                ax.set_title(title)
            ax.set_xlabel("Age")
            ax.set_ylabel("Component score (task-z)")
        fig.suptitle(feature.upper())
        fig.tight_layout()
        fig.savefig(out / f"stage3_age_scatter_{feature}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--components-out", required=True)
    parser.add_argument("--variance-out", required=True)
    parser.add_argument("--reliability-out", required=True)
    parser.add_argument("--age-models-out", required=True)
    parser.add_argument("--psych-models-out", required=True)
    parser.add_argument("--age-group-summary-out", required=True)
    parser.add_argument("--age-group-anova-out", required=True)
    parser.add_argument("--figures-dir", required=True)
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    metadata = pd.read_csv(args.metadata)
    features = features[features["task"].isin(PRIMARY_TASKS)].copy()

    components = build_components(features, metadata)
    variance = variance_decomposition(features)
    reliability = component_reliability(components)
    age_models, psych_models = fit_models(components)
    group_summary = age_group_summary(components)
    group_anova = age_group_anova(components)
    make_plots(components, args.figures_dir)

    for path, table in [
        (args.components_out, components),
        (args.variance_out, variance),
        (args.reliability_out, reliability),
        (args.age_models_out, age_models),
        (args.psych_models_out, psych_models),
        (args.age_group_summary_out, group_summary),
        (args.age_group_anova_out, group_anova),
    ]:
        ensure_parent(path)
        table.to_csv(path, index=False)


if __name__ == "__main__":
    main()

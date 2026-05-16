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


FEATURES = ["p1", "n1", "p3"]
FEATURE_LABELS = {"p1": "P1", "n1": "P2", "p3": "N450"}
COMPONENTS = ["component_general", "component_specific_sus_minus_ccd"]
COMPONENT_LABELS = {
    "component_general": "General",
    "component_specific_sus_minus_ccd": "Specific SuS-CCD",
}
COLORS = {"low": "#3775BA", "mean": "#767676", "high": "#B64342"}


def add_centered_terms(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["p_factor"] = pd.to_numeric(out["p_factor"], errors="coerce")
    age_mean = float(out["age"].mean())
    p_mean = float(out["p_factor"].mean())
    p_sd = float(out["p_factor"].std(ddof=1))
    out["age_c"] = out["age"] - age_mean
    out["age_c2"] = out["age_c"] ** 2
    out["p_c"] = out["p_factor"] - p_mean
    out["age_p"] = out["age_c"] * out["p_c"]
    out["age2_p"] = out["age_c2"] * out["p_c"]
    return out, {"age_mean": age_mean, "p_mean": p_mean, "p_sd": p_sd}


def fit_one(df: pd.DataFrame, outcome: str):
    model_df = df.dropna(subset=[outcome, "age", "sex", "p_factor"]).copy()
    if len(model_df) < 30:
        return None
    model_df, centers = add_centered_terms(model_df)
    base = smf.ols(f"{outcome} ~ age_c + age_c2 + C(sex) + p_c", data=model_df).fit()
    linear = smf.ols(f"{outcome} ~ age_c + age_c2 + C(sex) + p_c + age_c:p_c", data=model_df).fit()
    trajectory = smf.ols(
        f"{outcome} ~ age_c + age_c2 + C(sex) + p_c + age_c:p_c + age_c2:p_c",
        data=model_df,
    ).fit()
    return model_df, centers, base, linear, trajectory


def model_test_row(analysis: str, feature: str, component: str, test_name: str, base, model) -> dict:
    f_value, p_value, df_diff = model.compare_f_test(base)
    return {
        "analysis": analysis,
        "feature": feature,
        "component": component,
        "test": test_name,
        "n": int(model.nobs),
        "df_diff": float(df_diff),
        "f_value": float(f_value),
        "p_value": float(p_value),
        "base_r2": float(base.rsquared),
        "model_r2": float(model.rsquared),
        "delta_r2": float(model.rsquared - base.rsquared),
    }


def term_rows(analysis: str, feature: str, component: str, model_name: str, model) -> list[dict]:
    rows = []
    for term in ["p_c", "age_c:p_c", "age_c2:p_c"]:
        if term not in model.params:
            continue
        rows.append(
            {
                "analysis": analysis,
                "feature": feature,
                "component": component,
                "model": model_name,
                "term": term,
                "estimate": float(model.params[term]),
                "std_error": float(model.bse[term]),
                "t_value": float(model.tvalues[term]),
                "p_value": float(model.pvalues[term]),
                "r2": float(model.rsquared),
                "n": int(model.nobs),
            }
        )
    return rows


def run_analysis(analysis: str, components: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    tests: list[dict] = []
    terms: list[dict] = []
    fitted: dict = {}
    for feature in FEATURES:
        feature_df = components[components["feature"].eq(feature)].copy()
        for component in COMPONENTS:
            result = fit_one(feature_df, component)
            if result is None:
                continue
            model_df, centers, base, linear, trajectory = result
            tests.append(model_test_row(analysis, feature, component, "age_x_pfactor_linear_vs_base", base, linear))
            tests.append(model_test_row(analysis, feature, component, "age_and_age2_x_pfactor_vs_base", base, trajectory))
            terms.extend(term_rows(analysis, feature, component, "linear_interaction", linear))
            terms.extend(term_rows(analysis, feature, component, "trajectory_interaction", trajectory))
            fitted[(analysis, feature, component)] = {
                "model_df": model_df,
                "centers": centers,
                "base": base,
                "linear": linear,
                "trajectory": trajectory,
            }
    return pd.DataFrame(tests), pd.DataFrame(terms), fitted


def add_fdr(table: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = table.copy()
    if out.empty:
        return out
    out["q_value_global"] = multipletests(out["p_value"].fillna(1.0), method="fdr_bh")[1]
    out["q_value_within_group"] = np.nan
    for _, idx in out.groupby(group_cols).groups.items():
        values = out.loc[idx, "p_value"].fillna(1.0)
        out.loc[idx, "q_value_within_group"] = multipletests(values, method="fdr_bh")[1]
    return out


def prediction_grid(model_df: pd.DataFrame, centers: dict[str, float], model) -> pd.DataFrame:
    age_min = max(5.0, float(model_df["age"].min()))
    age_max = min(22.0, float(model_df["age"].max()))
    ages = np.linspace(age_min, age_max, 120)
    p_mean = centers["p_mean"]
    p_sd = centers["p_sd"]
    sex_mode = model_df["sex"].mode().iloc[0]
    rows = []
    for label, p_raw in [("low", p_mean - p_sd), ("mean", p_mean), ("high", p_mean + p_sd)]:
        tmp = pd.DataFrame({"age": ages, "p_factor": p_raw, "sex": sex_mode})
        tmp["age_c"] = tmp["age"] - centers["age_mean"]
        tmp["age_c2"] = tmp["age_c"] ** 2
        tmp["p_c"] = tmp["p_factor"] - p_mean
        tmp["age_p"] = tmp["age_c"] * tmp["p_c"]
        tmp["age2_p"] = tmp["age_c2"] * tmp["p_c"]
        pred = model.get_prediction(tmp).summary_frame(alpha=0.05)
        tmp["p_factor_level"] = label
        tmp["pred"] = pred["mean"].to_numpy()
        tmp["ci_low"] = pred["mean_ci_lower"].to_numpy()
        tmp["ci_high"] = pred["mean_ci_upper"].to_numpy()
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def plot_main_predictions(fitted: dict, tests: pd.DataFrame, out_base: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 7.2), sharex=True)
    for r, feature in enumerate(FEATURES):
        for c, component in enumerate(COMPONENTS):
            ax = axes[r, c]
            payload = fitted.get(("main", feature, component))
            if payload is None:
                ax.axis("off")
                continue
            model_df = payload["model_df"]
            pred = prediction_grid(model_df, payload["centers"], payload["trajectory"])
            rng = np.random.default_rng(20260511 + r * 10 + c)
            sample = model_df.sample(n=min(len(model_df), 250), random_state=20260511)
            ax.scatter(
                sample["age"],
                sample[component],
                s=6,
                alpha=0.16,
                color="#767676",
                linewidths=0,
            )
            for label, display in [("low", "-1 SD p-factor"), ("mean", "Mean"), ("high", "+1 SD p-factor")]:
                sub = pred[pred["p_factor_level"].eq(label)]
                ax.plot(sub["age"], sub["pred"], color=COLORS[label], linewidth=1.4, label=display)
                ax.fill_between(sub["age"].to_numpy(), sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy(), color=COLORS[label], alpha=0.08)
            row = tests[
                (tests["analysis"].eq("main"))
                & (tests["feature"].eq(feature))
                & (tests["component"].eq(component))
                & (tests["test"].eq("age_and_age2_x_pfactor_vs_base"))
            ]
            q_text = ""
            if not row.empty:
                q_text = f"interaction q={row.iloc[0]['q_value_within_group']:.3g}"
            ax.text(0.02, 0.97, q_text, transform=ax.transAxes, ha="left", va="top", fontsize=6.2)
            ax.axhline(0, color="#D8D8D8", linewidth=0.7)
            ax.set_title(f"{FEATURE_LABELS[feature]} {COMPONENT_LABELS[component]}")
            if r == 2:
                ax.set_xlabel("Age (years)")
            if c == 0:
                ax.set_ylabel("Component score")
            if r == 0 and c == 1:
                ax.legend(loc="upper right", fontsize=6, frameon=False)
    fig.suptitle("Predicted developmental trajectories by p-factor level", y=0.995, fontsize=9, fontweight="bold")
    fig.tight_layout(h_pad=1.1, w_pad=1.2)
    ensure_parent(str(out_base))
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-components", required=True)
    parser.add_argument("--high-quality-components", required=True)
    parser.add_argument("--full-pheno-components", required=True)
    parser.add_argument("--high-quality-full-pheno-components", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    args = parser.parse_args()

    sources = {
        "main": pd.read_csv(args.main_components),
        "high_quality": pd.read_csv(args.high_quality_components),
        "full_pheno": pd.read_csv(args.full_pheno_components),
        "high_quality_full_pheno": pd.read_csv(args.high_quality_full_pheno_components),
    }

    all_tests = []
    all_terms = []
    all_fitted = {}
    for analysis, table in sources.items():
        tests, terms, fitted = run_analysis(analysis, table)
        all_tests.append(tests)
        all_terms.append(terms)
        all_fitted.update(fitted)

    tests = pd.concat(all_tests, ignore_index=True)
    terms = pd.concat(all_terms, ignore_index=True)
    tests = add_fdr(tests, ["analysis", "test"])
    terms = add_fdr(terms, ["analysis", "model", "term"])

    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tests.to_csv(out_dir / "pfactor_age_interaction_model_tests.csv", index=False)
    terms.to_csv(out_dir / "pfactor_age_interaction_terms.csv", index=False)

    plot_main_predictions(
        all_fitted,
        tests,
        figures_dir / "pfactor_age_interaction_main_predictions",
    )

    sample_rows = []
    for analysis, table in sources.items():
        unique = table.drop_duplicates("subject")
        complete = unique.dropna(subset=["age", "sex", "p_factor"])
        sample_rows.append(
            {
                "analysis": analysis,
                "subjects_total": int(unique["subject"].nunique()),
                "subjects_with_age_sex_pfactor": int(complete["subject"].nunique()),
                "mean_age": float(pd.to_numeric(complete["age"], errors="coerce").mean()),
                "mean_p_factor": float(pd.to_numeric(complete["p_factor"], errors="coerce").mean()),
            }
        )
    pd.DataFrame(sample_rows).to_csv(out_dir / "pfactor_age_interaction_sample_summary.csv", index=False)

    print(tests.sort_values(["analysis", "test", "q_value_within_group"]).to_string(index=False))


if __name__ == "__main__":
    main()

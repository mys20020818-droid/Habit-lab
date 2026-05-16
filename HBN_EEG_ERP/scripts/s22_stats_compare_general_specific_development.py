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
FEATURE_LABELS = {"p1": "P1", "n1": "P2", "p3": "N450"}
COMPONENTS = ["component_general", "component_specific_sus_minus_ccd"]
COMPONENT_LABELS = {
    "component_general": "General",
    "component_specific_sus_minus_ccd": "Specific SuS-CCD",
}
COMPONENT_SHORT = {
    "component_general": "general",
    "component_specific_sus_minus_ccd": "specific",
}
COLORS = {"general": "#0F4D92", "specific": "#E3A23A"}


def centered(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    out = df.copy()
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    age_mean = float(out["age"].mean())
    out["age_c"] = out["age"] - age_mean
    out["age_c2"] = out["age_c"] ** 2
    return out, age_mean


def ftest_nested(reduced, full) -> tuple[float, float, float]:
    f_value, p_value, df_diff = full.compare_f_test(reduced)
    return float(f_value), float(p_value), float(df_diff)


def curve_range(model, age_mean: float, sex_mode: str, age_min: float = 5.0, age_max: float = 21.0) -> tuple[float, float, float]:
    ages = np.linspace(age_min, age_max, 200)
    pred_df = pd.DataFrame({"age": ages, "sex": sex_mode})
    pred_df["age_c"] = pred_df["age"] - age_mean
    pred_df["age_c2"] = pred_df["age_c"] ** 2
    pred = model.predict(pred_df)
    return float(pred.iloc[-1] - pred.iloc[0]), float(pred.max() - pred.min()), float(ages[int(np.argmax(pred.to_numpy()))])


def component_age_models(components: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for feature in FEATURES:
        fdf = components[components["feature"].eq(feature)].copy()
        for component in COMPONENTS:
            df = fdf.dropna(subset=[component, "age", "sex"]).copy()
            if len(df) < 30:
                continue
            df, age_mean = centered(df)
            sex_mode = str(df["sex"].mode().iloc[0])
            sex_model = smf.ols(f"{component} ~ C(sex)", data=df).fit()
            linear_model = smf.ols(f"{component} ~ age_c + C(sex)", data=df).fit()
            quad_model = smf.ols(f"{component} ~ age_c + age_c2 + C(sex)", data=df).fit()
            f_age, p_age, df_age = ftest_nested(sex_model, quad_model)
            f_linear, p_linear, df_linear = ftest_nested(sex_model, linear_model)
            f_quad, p_quad, df_quad = ftest_nested(linear_model, quad_model)
            change_5_21, range_5_21, peak_age = curve_range(quad_model, age_mean, sex_mode)
            rows.append(
                {
                    "feature": feature,
                    "component": component,
                    "component_short": COMPONENT_SHORT[component],
                    "n": int(quad_model.nobs),
                    "r2_sex_only": float(sex_model.rsquared),
                    "r2_linear_age": float(linear_model.rsquared),
                    "r2_quadratic_age": float(quad_model.rsquared),
                    "delta_r2_age_total": float(quad_model.rsquared - sex_model.rsquared),
                    "delta_r2_linear": float(linear_model.rsquared - sex_model.rsquared),
                    "delta_r2_quadratic_extra": float(quad_model.rsquared - linear_model.rsquared),
                    "f_age_total": f_age,
                    "p_age_total": p_age,
                    "df_age_total": df_age,
                    "f_linear_age": f_linear,
                    "p_linear_age": p_linear,
                    "f_quadratic_extra": f_quad,
                    "p_quadratic_extra": p_quad,
                    "age_linear_estimate": float(quad_model.params.get("age_c", np.nan)),
                    "age_linear_p": float(quad_model.pvalues.get("age_c", np.nan)),
                    "age_quadratic_estimate": float(quad_model.params.get("age_c2", np.nan)),
                    "age_quadratic_p": float(quad_model.pvalues.get("age_c2", np.nan)),
                    "predicted_change_5_to_21": change_5_21,
                    "predicted_range_5_to_21": range_5_21,
                    "peak_or_trough_age_if_quadratic": peak_age,
                }
            )
    out = pd.DataFrame(rows)
    for col in ["p_age_total", "p_linear_age", "p_quadratic_extra", "age_linear_p", "age_quadratic_p"]:
        out[f"q_{col.removeprefix('p_')}"] = multipletests(out[col].fillna(1.0), method="fdr_bh")[1]
    return out


def build_long(components: pd.DataFrame, zscore_within_component: bool = False) -> pd.DataFrame:
    rows = []
    for component in COMPONENTS:
        sub = components[
            ["subject", "feature", "age", "sex", component]
        ].rename(columns={component: "score"}).copy()
        sub["component_type"] = COMPONENT_SHORT[component]
        rows.append(sub)
    long = pd.concat(rows, ignore_index=True).dropna(subset=["score", "age", "sex"])
    if zscore_within_component:
        long["score"] = long.groupby(["feature", "component_type"])["score"].transform(
            lambda s: (s - s.mean()) / s.std(ddof=1)
        )
    return long


def component_difference_tests(components: pd.DataFrame, zscore_within_component: bool = False) -> pd.DataFrame:
    long = build_long(components, zscore_within_component=zscore_within_component)
    rows: list[dict] = []
    scale = "within_component_z" if zscore_within_component else "raw_component_score"
    for feature in FEATURES:
        df = long[long["feature"].eq(feature)].copy()
        if len(df) < 60:
            continue
        df, age_mean = centered(df)
        df["component_type"] = pd.Categorical(df["component_type"], categories=["general", "specific"])
        model = smf.ols(
            "score ~ C(component_type) * age_c + C(component_type) * age_c2 + C(sex)",
            data=df,
        ).fit(cov_type="cluster", cov_kwds={"groups": df["subject"]})
        linear_term = "C(component_type)[T.specific]:age_c"
        quad_term = "C(component_type)[T.specific]:age_c2"
        joint = model.f_test(f"{linear_term} = 0, {quad_term} = 0")
        rows.append(
            {
                "feature": feature,
                "scale": scale,
                "n_rows": int(model.nobs),
                "n_subjects": int(df["subject"].nunique()),
                "general_age_linear_estimate": float(model.params.get("age_c", np.nan)),
                "specific_minus_general_age_linear_estimate": float(model.params.get(linear_term, np.nan)),
                "specific_age_linear_estimate": float(
                    model.params.get("age_c", np.nan) + model.params.get(linear_term, 0.0)
                ),
                "linear_shape_difference_p": float(model.pvalues.get(linear_term, np.nan)),
                "general_age_quadratic_estimate": float(model.params.get("age_c2", np.nan)),
                "specific_minus_general_age_quadratic_estimate": float(model.params.get(quad_term, np.nan)),
                "specific_age_quadratic_estimate": float(
                    model.params.get("age_c2", np.nan) + model.params.get(quad_term, 0.0)
                ),
                "quadratic_shape_difference_p": float(model.pvalues.get(quad_term, np.nan)),
                "joint_shape_difference_f": float(np.ravel(joint.fvalue)[0]),
                "joint_shape_difference_p": float(joint.pvalue),
                "model_r2": float(model.rsquared),
            }
        )
    out = pd.DataFrame(rows)
    for scale_name, idx in out.groupby("scale").groups.items():
        out.loc[idx, "q_joint_shape_difference"] = multipletests(
            out.loc[idx, "joint_shape_difference_p"].fillna(1.0), method="fdr_bh"
        )[1]
        out.loc[idx, "q_linear_shape_difference"] = multipletests(
            out.loc[idx, "linear_shape_difference_p"].fillna(1.0), method="fdr_bh"
        )[1]
        out.loc[idx, "q_quadratic_shape_difference"] = multipletests(
            out.loc[idx, "quadratic_shape_difference_p"].fillna(1.0), method="fdr_bh"
        )[1]
    return out


def component_contrast_summary(models: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for feature in FEATURES:
        g = models[(models["feature"].eq(feature)) & (models["component_short"].eq("general"))].iloc[0]
        s = models[(models["feature"].eq(feature)) & (models["component_short"].eq("specific"))].iloc[0]
        rows.append(
            {
                "feature": feature,
                "general_delta_r2_age": float(g["delta_r2_age_total"]),
                "specific_delta_r2_age": float(s["delta_r2_age_total"]),
                "general_to_specific_delta_r2_ratio": float(g["delta_r2_age_total"] / s["delta_r2_age_total"])
                if float(s["delta_r2_age_total"]) > 0
                else np.inf,
                "general_predicted_range_5_21": float(g["predicted_range_5_to_21"]),
                "specific_predicted_range_5_21": float(s["predicted_range_5_to_21"]),
                "general_to_specific_range_ratio": float(g["predicted_range_5_to_21"] / s["predicted_range_5_to_21"])
                if float(s["predicted_range_5_to_21"]) > 0
                else np.inf,
                "general_quadratic_extra_delta_r2": float(g["delta_r2_quadratic_extra"]),
                "specific_quadratic_extra_delta_r2": float(s["delta_r2_quadratic_extra"]),
                "general_q_age_total": float(g["q_age_total"]),
                "specific_q_age_total": float(s["q_age_total"]),
            }
        )
    return pd.DataFrame(rows)


def predictions_for_plot(components: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        for component in COMPONENTS:
            df = components[components["feature"].eq(feature)].dropna(subset=[component, "age", "sex"]).copy()
            df, age_mean = centered(df)
            sex_mode = str(df["sex"].mode().iloc[0])
            model = smf.ols(f"{component} ~ age_c + age_c2 + C(sex)", data=df).fit()
            ages = np.linspace(5, 21, 150)
            pred_df = pd.DataFrame({"age": ages, "sex": sex_mode})
            pred_df["age_c"] = pred_df["age"] - age_mean
            pred_df["age_c2"] = pred_df["age_c"] ** 2
            pred = model.get_prediction(pred_df).summary_frame(alpha=0.05)
            for i, age in enumerate(ages):
                rows.append(
                    {
                        "feature": feature,
                        "component": COMPONENT_SHORT[component],
                        "age": float(age),
                        "pred": float(pred.iloc[i]["mean"]),
                        "ci_low": float(pred.iloc[i]["mean_ci_lower"]),
                        "ci_high": float(pred.iloc[i]["mean_ci_upper"]),
                    }
                )
    return pd.DataFrame(rows)


def plot_comparison(components: pd.DataFrame, models: pd.DataFrame, contrast: pd.DataFrame, out_base: Path) -> None:
    pred = predictions_for_plot(components)
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.8), gridspec_kw={"height_ratios": [1.05, 0.9]})

    for c, feature in enumerate(FEATURES):
        ax = axes[0, c]
        rng = np.random.default_rng(20260511 + c)
        fdf = components[components["feature"].eq(feature)].dropna(subset=["age"])
        y_values = pd.concat(
            [
                pd.to_numeric(fdf["component_general"], errors="coerce"),
                pd.to_numeric(fdf["component_specific_sus_minus_ccd"], errors="coerce"),
            ],
            ignore_index=True,
        ).dropna()
        y_lo, y_hi = np.percentile(y_values, [1, 99])
        y_pad = (y_hi - y_lo) * 0.12 if y_hi > y_lo else 0.5
        sample = fdf.sample(n=min(len(fdf), 350), random_state=20260511)
        for component, color, alpha in [
            ("component_general", COLORS["general"], 0.16),
            ("component_specific_sus_minus_ccd", COLORS["specific"], 0.13),
        ]:
            jitter = rng.normal(0, 0.035, size=len(sample))
            ax.scatter(sample["age"] + jitter, sample[component], s=7, color=color, alpha=alpha, linewidths=0)
        for component, color, label in [
            ("general", COLORS["general"], "General"),
            ("specific", COLORS["specific"], "Specific"),
        ]:
            sub = pred[(pred["feature"].eq(feature)) & (pred["component"].eq(component))]
            ax.plot(sub["age"], sub["pred"], color=color, linewidth=1.6, label=label)
            ax.fill_between(sub["age"].to_numpy(), sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy(), color=color, alpha=0.08)
        raw_test = contrast[(contrast["feature"].eq(feature)) & (contrast["scale"].eq("raw_component_score"))].iloc[0]
        ax.text(
            0.02,
            0.97,
            f"shape diff q={raw_test['q_joint_shape_difference']:.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
        ax.axhline(0, color="#D8D8D8", linewidth=0.7)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        ax.set_title(f"{FEATURE_LABELS[feature]} developmental curves")
        ax.set_xlabel("Age (years)")
        if c == 0:
            ax.set_ylabel("Component score")
        if c == 2:
            ax.legend(loc="upper left", fontsize=6, frameon=False)

    ax = axes[1, 0]
    x = np.arange(len(FEATURES))
    width = 0.34
    g = models[models["component_short"].eq("general")].set_index("feature").loc[FEATURES]
    s = models[models["component_short"].eq("specific")].set_index("feature").loc[FEATURES]
    ax.bar(x - width / 2, g["delta_r2_age_total"], width=width, color=COLORS["general"], label="General")
    ax.bar(x + width / 2, s["delta_r2_age_total"], width=width, color=COLORS["specific"], label="Specific")
    ax.set_xticks(x, [FEATURE_LABELS[f] for f in FEATURES])
    ax.set_ylabel("Age effect ΔR2")
    ax.set_title("Effect magnitude")
    ax.set_ylabel("Age effect Delta R2")
    ax.legend(loc="upper left", fontsize=6, frameon=False)

    ax = axes[1, 1]
    ax.bar(x - width / 2, g["delta_r2_linear"], width=width, color=COLORS["general"], label="Linear")
    ax.bar(x + width / 2, g["delta_r2_quadratic_extra"], width=width, color="#8FB8DE", label="Quadratic extra")
    ax.set_xticks(x, [FEATURE_LABELS[f] for f in FEATURES])
    ax.set_ylabel("General ΔR2")
    ax.set_title("General shape")
    ax.set_ylabel("General Delta R2")
    ax.legend(loc="upper left", fontsize=6, frameon=False)

    ax = axes[1, 2]
    ax.bar(x - width / 2, s["delta_r2_linear"], width=width, color=COLORS["specific"], label="Linear")
    ax.bar(x + width / 2, s["delta_r2_quadratic_extra"], width=width, color="#F0D6A2", label="Quadratic extra")
    ax.set_xticks(x, [FEATURE_LABELS[f] for f in FEATURES])
    ax.set_ylabel("Specific ΔR2")
    ax.set_title("Specific shape")
    ax.set_ylabel("Specific Delta R2")
    ax.legend(loc="upper left", fontsize=6, frameon=False)

    for top_ax in axes[0, :]:
        legend = top_ax.get_legend()
        if legend is not None:
            legend.remove()
    handles = [
        plt.Line2D([0], [0], color=COLORS["general"], linewidth=1.8, label="General"),
        plt.Line2D([0], [0], color=COLORS["specific"], linewidth=1.8, label="Specific"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Task-general and task-specific components show separable developmental profiles", y=1.02, fontsize=9, fontweight="bold")
    for legend in list(fig.legends):
        legend.remove()
    fig.tight_layout(h_pad=1.1, w_pad=1.1)
    ensure_parent(str(out_base))
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    args = parser.parse_args()

    components = pd.read_csv(args.components)
    models = component_age_models(components)
    contrast_raw = component_difference_tests(components, zscore_within_component=False)
    contrast_z = component_difference_tests(components, zscore_within_component=True)
    contrast = pd.concat([contrast_raw, contrast_z], ignore_index=True)
    summary = component_contrast_summary(models)

    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    models.to_csv(out_dir / "general_specific_age_curve_models.csv", index=False)
    contrast.to_csv(out_dir / "general_specific_shape_difference_tests.csv", index=False)
    summary.to_csv(out_dir / "general_specific_development_contrast_summary.csv", index=False)
    plot_comparison(components, models, contrast, figures_dir / "general_specific_development_comparison")

    print(summary.to_string(index=False))
    print(contrast.sort_values(["scale", "joint_shape_difference_p"]).to_string(index=False))


if __name__ == "__main__":
    main()

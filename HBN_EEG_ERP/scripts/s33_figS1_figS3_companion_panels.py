from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures" / "nature_companion"
SOURCE_DIR = TABLE_DIR / "nature_companion"
REPORT_PATH = ROOT / "results" / "reports" / "nature_companion_figures_report.md"

AGE_GROUPS = ["child_5_9", "adolescent_10_14", "youth_15_21"]
AGE_LABELS = {
    "child_5_9": "5-9 y",
    "adolescent_10_14": "10-14 y",
    "youth_15_21": "15-21 y",
}
AGE_COLORS = {
    "child_5_9": "#5DA5DA",
    "adolescent_10_14": "#72B36A",
    "youth_15_21": "#D99245",
}
FEATURES = ["p1", "n1", "p3"]
FEATURE_LABELS = {"p1": "P1", "n1": "P2", "p3": "N450"}
COMPONENTS = ["component_general", "component_specific_sus_minus_ccd"]
COMPONENT_SHORT = {
    "component_general": "General",
    "component_specific_sus_minus_ccd": "Specific",
    "general": "General",
    "specific_sus_minus_ccd": "Specific",
}
COMPONENT_COLORS = {
    "component_general": "#303030",
    "component_specific_sus_minus_ccd": "#6A5ACD",
    "general": "#303030",
    "specific_sus_minus_ccd": "#6A5ACD",
}
TASK_COLORS = {"CCD": "#597DBE", "SuS": "#2A9D8F"}
STATUS_COLORS = {"usable": "#4C9F70", "zero kept": "#D7A33F", "failed": "#9A9A9A"}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_pub(fig, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{base_path}.svg", bbox_inches="tight")
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base_path}.tiff", dpi=600, bbox_inches="tight")


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def minus_log10(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr, 1e-300, 1.0)
    return -np.log10(arr)


def pretty_component(component: str) -> str:
    return COMPONENT_SHORT.get(component, component)


def read_table(relative: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / relative)


def prepare_sample_sources() -> dict[str, pd.DataFrame]:
    metadata = read_table("downloaded_subjects_age_groups.csv")
    selected = read_table("selected_electrodes/selected_electrode_erp_features_by_subject.csv")
    decomp = read_table("full_sus_ccd_decomp_subjects.csv")
    qc = read_table("full_sus_ccd_epoch_qc_analysisch_reject250.csv")
    p_interaction = read_table("pfactor_age_interaction/pfactor_age_interaction_model_tests.csv")

    metadata = numeric(metadata, ["age", "p_factor"])
    qc = numeric(qc, ["n_epochs_kept", "n_visual_events", "drop_fraction", "p2p_uv_median", "p2p_uv_p90"])

    final_subjects = selected["subject"].drop_duplicates()
    final_meta = metadata[metadata["subject"].isin(final_subjects)].copy()
    final_meta = final_meta[final_meta["age_group"].isin(AGE_GROUPS)].copy()

    downloaded_5_21 = metadata[metadata["age_group"].isin(AGE_GROUPS)]["subject"].nunique()
    p_overlap = int(pd.to_numeric(p_interaction["n"], errors="coerce").dropna().max())
    sample_flow = pd.DataFrame(
        [
            {"stage": "Downloaded age 5-21", "subjects": int(downloaded_5_21)},
            {"stage": "SuS+CCD components", "subjects": int(decomp["subject"].nunique())},
            {"stage": "Full-channel selected ERP", "subjects": int(final_subjects.nunique())},
            {"stage": "p-factor overlap", "subjects": p_overlap},
        ]
    )

    age_sex = (
        final_meta.assign(sex=final_meta["sex"].fillna("NA"))
        .groupby(["age_group", "sex"], observed=False)["subject"]
        .nunique()
        .reset_index(name="n_subjects")
    )
    age_sex["age_group"] = pd.Categorical(age_sex["age_group"], AGE_GROUPS, ordered=True)
    age_sex = age_sex.sort_values(["age_group", "sex"])

    qc["qc_class"] = "usable"
    qc.loc[qc["status"].ne("ok"), "qc_class"] = "failed"
    qc.loc[qc["status"].eq("ok") & (qc["n_epochs_kept"].fillna(0) <= 0), "qc_class"] = "zero kept"
    qc_summary = (
        qc.groupby(["task", "qc_class"], observed=False)
        .size()
        .reset_index(name="n_recordings")
    )
    qc_summary["task"] = pd.Categorical(qc_summary["task"], ["CCD", "SuS"], ordered=True)
    qc_summary["qc_class"] = pd.Categorical(qc_summary["qc_class"], ["usable", "zero kept", "failed"], ordered=True)
    qc_summary = qc_summary.sort_values(["task", "qc_class"])

    epoch_summary = (
        qc[qc["qc_class"].eq("usable")]
        .groupby("task", observed=False)
        .agg(
            n_recordings=("subject", "size"),
            median_epochs_kept=("n_epochs_kept", "median"),
            mean_epochs_kept=("n_epochs_kept", "mean"),
            median_drop_fraction=("drop_fraction", "median"),
            p90_drop_fraction=("drop_fraction", lambda x: float(np.nanpercentile(x, 90))),
        )
        .reset_index()
    )

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    sample_flow.to_csv(SOURCE_DIR / "sample_flow.csv", index=False)
    age_sex.to_csv(SOURCE_DIR / "analysis_subject_age_sex_counts.csv", index=False)
    qc_summary.to_csv(SOURCE_DIR / "recording_qc_summary.csv", index=False)
    epoch_summary.to_csv(SOURCE_DIR / "usable_recording_epoch_summary.csv", index=False)

    return {
        "metadata": metadata,
        "final_meta": final_meta,
        "qc": qc,
        "sample_flow": sample_flow,
        "age_sex": age_sex,
        "qc_summary": qc_summary,
        "epoch_summary": epoch_summary,
    }


def plot_sample_qc_figure(sources: dict[str, pd.DataFrame]) -> Path:
    final_meta = sources["final_meta"]
    qc = sources["qc"]
    sample_flow = sources["sample_flow"]
    age_sex = sources["age_sex"]
    qc_summary = sources["qc_summary"]

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.6))
    axes = axes.ravel()

    ax = axes[0]
    y = np.arange(len(sample_flow))[::-1]
    ax.barh(y, sample_flow["subjects"], color=["#D7D7D7", "#9ECAE1", "#4F93C4", "#222222"], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(sample_flow["stage"])
    ax.set_xlabel("Subjects")
    ax.set_title("Analysis flow")
    ax.set_xlim(0, sample_flow["subjects"].max() * 1.16)
    for yi, count in zip(y, sample_flow["subjects"]):
        ax.text(count + sample_flow["subjects"].max() * 0.02, yi, f"{int(count):,}", va="center", fontsize=7)
    panel_label(ax, "a")

    ax = axes[1]
    bins = np.arange(5, 22.5, 0.5)
    hist_data = [
        final_meta.loc[final_meta["age_group"].eq(age_group), "age"].dropna().to_numpy()
        for age_group in AGE_GROUPS
    ]
    ax.hist(
        hist_data,
        bins=bins,
        stacked=True,
        color=[AGE_COLORS[g] for g in AGE_GROUPS],
        label=[AGE_LABELS[g] for g in AGE_GROUPS],
        edgecolor="white",
        linewidth=0.25,
    )
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Subjects")
    ax.set_title("Age distribution")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    panel_label(ax, "b")

    ax = axes[2]
    pivot = age_sex.pivot_table(index="age_group", columns="sex", values="n_subjects", fill_value=0, observed=False)
    sex_order = [sex for sex in ["F", "M", "NA"] if sex in pivot.columns] + [sex for sex in pivot.columns if sex not in {"F", "M", "NA"}]
    bottom = np.zeros(len(AGE_GROUPS))
    sex_colors = {"F": "#C76782", "M": "#5276A7", "NA": "#B7B7B7"}
    for sex in sex_order:
        vals = pivot.reindex(AGE_GROUPS)[sex].to_numpy(dtype=float)
        ax.bar(np.arange(len(AGE_GROUPS)), vals, bottom=bottom, color=sex_colors.get(sex, "#8C8C8C"), width=0.68, label=sex)
        bottom += vals
    ax.set_xticks(np.arange(len(AGE_GROUPS)))
    ax.set_xticklabels([AGE_LABELS[g] for g in AGE_GROUPS])
    ax.set_ylabel("Subjects")
    ax.set_title("Sex by age group")
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper right")
    panel_label(ax, "c")

    ax = axes[3]
    status_pivot = qc_summary.pivot_table(index="task", columns="qc_class", values="n_recordings", fill_value=0, observed=False)
    tasks = ["CCD", "SuS"]
    bottom = np.zeros(len(tasks))
    for status in ["usable", "zero kept", "failed"]:
        vals = status_pivot.reindex(tasks).get(status, pd.Series(0, index=tasks)).to_numpy(dtype=float)
        ax.bar(np.arange(len(tasks)), vals, bottom=bottom, color=STATUS_COLORS[status], width=0.58, label=status)
        bottom += vals
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Recordings")
    ax.set_title("Recording QC")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    panel_label(ax, "d")

    ax = axes[4]
    usable = qc[qc["qc_class"].eq("usable")].copy()
    box_data = [usable.loc[usable["task"].eq(task), "n_epochs_kept"].dropna().to_numpy() for task in tasks]
    bp = ax.boxplot(box_data, tick_labels=tasks, patch_artist=True, showfliers=False, widths=0.5)
    for patch, task in zip(bp["boxes"], tasks):
        patch.set_facecolor(TASK_COLORS[task])
        patch.set_alpha(0.82)
        patch.set_edgecolor("#444444")
    for key in ["medians", "whiskers", "caps"]:
        for item in bp[key]:
            item.set_color("#333333")
            item.set_linewidth(0.75)
    ax.set_ylabel("Kept visual epochs")
    ax.set_title("Usable epochs")
    for idx, task in enumerate(tasks, start=1):
        n = usable.loc[usable["task"].eq(task), "n_epochs_kept"].notna().sum()
        med = usable.loc[usable["task"].eq(task), "n_epochs_kept"].median()
        ax.text(idx, ax.get_ylim()[1] * 0.96, f"n={n:,}\nmed={med:.0f}", ha="center", va="top", fontsize=6)
    panel_label(ax, "e")

    ax = axes[5]
    box_data = [usable.loc[usable["task"].eq(task), "drop_fraction"].dropna().to_numpy() * 100 for task in tasks]
    bp = ax.boxplot(box_data, tick_labels=tasks, patch_artist=True, showfliers=False, widths=0.5)
    for patch, task in zip(bp["boxes"], tasks):
        patch.set_facecolor(TASK_COLORS[task])
        patch.set_alpha(0.82)
        patch.set_edgecolor("#444444")
    for key in ["medians", "whiskers", "caps"]:
        for item in bp[key]:
            item.set_color("#333333")
            item.set_linewidth(0.75)
    ax.set_ylabel("Dropped epochs (%)")
    ax.set_title("Artifact rejection burden")
    panel_label(ax, "f")

    fig.suptitle("Sample coverage and EEG quality control", y=0.99, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = FIG_DIR / "figS1"
    save_pub(fig, out)
    plt.close(fig)
    return out


def prepare_development_sources() -> dict[str, pd.DataFrame]:
    reliability = read_table("full_sus_ccd_split_half_reliability.csv")
    variance = read_table("full_sus_ccd_variance_decomposition.csv")
    contrast = read_table("general_specific_development/general_specific_development_contrast_summary.csv")
    shape = read_table("general_specific_development/general_specific_shape_difference_tests.csv")
    replication = read_table("full_sus_ccd_followup/full_sus_ccd_core_age_replication.csv")

    reliability = numeric(reliability, ["spearman_brown", "pearson_r_odd_even", "mean_n_epochs"])
    variance = numeric(variance, ["icc_general", "icc_specific"])
    contrast = numeric(
        contrast,
        [
            "general_delta_r2_age",
            "specific_delta_r2_age",
            "general_to_specific_delta_r2_ratio",
            "general_to_specific_range_ratio",
            "general_q_age_total",
            "specific_q_age_total",
        ],
    )
    shape = numeric(
        shape,
        [
            "joint_shape_difference_p",
            "q_joint_shape_difference",
            "q_linear_shape_difference",
            "q_quadratic_shape_difference",
            "model_r2",
        ],
    )
    replication = numeric(replication, ["n", "f_value", "p_value", "q_value", "r_squared"])

    reliability.to_csv(SOURCE_DIR / "development_reliability_source.csv", index=False)
    variance.to_csv(SOURCE_DIR / "development_variance_source.csv", index=False)
    contrast.to_csv(SOURCE_DIR / "development_delta_r2_source.csv", index=False)
    shape.to_csv(SOURCE_DIR / "development_shape_difference_source.csv", index=False)
    replication.to_csv(SOURCE_DIR / "development_replication_source.csv", index=False)

    return {
        "reliability": reliability,
        "variance": variance,
        "contrast": contrast,
        "shape": shape,
        "replication": replication,
    }


def plot_development_figure(sources: dict[str, pd.DataFrame]) -> Path:
    reliability = sources["reliability"]
    variance = sources["variance"]
    contrast = sources["contrast"]
    shape = sources["shape"]
    replication = sources["replication"]

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.5))
    axes = axes.ravel()
    x = np.arange(len(FEATURES))

    ax = axes[0]
    width = 0.34
    for i, task in enumerate(["CCD", "SuS"]):
        vals = []
        for feature in FEATURES:
            row = reliability[reliability["task"].eq(task) & reliability["feature"].eq(feature)]
            vals.append(float(row["spearman_brown"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + (i - 0.5) * width, vals, width=width, color=TASK_COLORS[task], label=task)
    ax.axhline(0.70, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURES])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Spearman-Brown")
    ax.set_title("Split-half reliability")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    panel_label(ax, "a")

    ax = axes[1]
    general = variance.set_index("feature").reindex(FEATURES)["icc_general"].to_numpy(dtype=float)
    specific = variance.set_index("feature").reindex(FEATURES)["icc_specific"].to_numpy(dtype=float)
    ax.bar(x, general, color=COMPONENT_COLORS["component_general"], width=0.62, label="General")
    ax.bar(x, specific, bottom=general, color=COMPONENT_COLORS["component_specific_sus_minus_ccd"], width=0.62, label="Specific")
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURES])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Variance share")
    ax.set_title("Task-general vs specific variance")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    panel_label(ax, "b")

    ax = axes[2]
    general = contrast.set_index("feature").reindex(FEATURES)["general_delta_r2_age"].to_numpy(dtype=float) * 100
    specific = contrast.set_index("feature").reindex(FEATURES)["specific_delta_r2_age"].to_numpy(dtype=float) * 100
    ax.bar(x - width / 2, general, width=width, color=COMPONENT_COLORS["component_general"], label="General")
    ax.bar(x + width / 2, specific, width=width, color=COMPONENT_COLORS["component_specific_sus_minus_ccd"], label="Specific")
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURES])
    ax.set_ylabel("Age delta R2 (%)")
    ax.set_title("Developmental effect size")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    panel_label(ax, "c")

    ax = axes[3]
    ratio_delta = contrast.set_index("feature").reindex(FEATURES)["general_to_specific_delta_r2_ratio"].to_numpy(dtype=float)
    ratio_range = contrast.set_index("feature").reindex(FEATURES)["general_to_specific_range_ratio"].to_numpy(dtype=float)
    ax.plot(x, ratio_delta, marker="o", color="#303030", label="Delta R2 ratio", linewidth=1.2)
    ax.plot(x, ratio_range, marker="s", color="#A05A9C", label="Predicted range ratio", linewidth=1.2)
    ax.axhline(1, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURES])
    ax.set_ylabel("General / specific")
    ax.set_title("Effect-size separation")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    panel_label(ax, "d")

    ax = axes[4]
    shape_raw = shape[shape["scale"].eq("raw_component_score")].set_index("feature").reindex(FEATURES)
    qvals = shape_raw["q_joint_shape_difference"].to_numpy(dtype=float)
    ax.bar(x, minus_log10(qvals), color=["#9A9A9A", "#D15F5F", "#D15F5F"], width=0.62)
    ax.axhline(-np.log10(0.05), color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURES])
    ax.set_ylabel("-log10(q)")
    ax.set_title("General-specific shape difference")
    panel_label(ax, "e")

    ax = axes[5]
    analysis_order = ["main", "high_quality", "full_pheno", "high_quality_full_pheno"]
    analysis_labels = ["Main", "HQ", "Pheno", "HQ+pheno"]
    for feature, color in [("n1", "#5A6FAE"), ("p3", "#C26C4A")]:
        sub = (
            replication[
                replication["feature"].eq(feature)
                & replication["component"].eq("component_general")
                & replication["analysis"].isin(analysis_order)
            ]
            .set_index("analysis")
            .reindex(analysis_order)
        )
        ax.plot(np.arange(len(analysis_order)), sub["r_squared"].to_numpy(dtype=float) * 100, marker="o", linewidth=1.3, color=color, label=FEATURE_LABELS[feature])
        for xi, (_, row) in enumerate(sub.iterrows()):
            if np.isfinite(row["n"]):
                ax.text(xi, row["r_squared"] * 100 + 0.8, f"n={int(row['n'])}", ha="center", fontsize=5.6, color=color)
    ax.set_xticks(np.arange(len(analysis_order)))
    ax.set_xticklabels(analysis_labels)
    ax.set_ylabel("Age model R2 (%)")
    ax.set_title("Core age effect replication")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    panel_label(ax, "f")

    fig.suptitle("Reliability and developmental separability", y=0.99, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = FIG_DIR / "figS2"
    save_pub(fig, out)
    plt.close(fig)
    return out


def prepare_clinical_sources() -> dict[str, pd.DataFrame]:
    pf_tests = read_table("pfactor_age_interaction/pfactor_age_interaction_model_tests.csv")
    pf_terms = read_table("pfactor_age_interaction/pfactor_age_interaction_terms.csv")
    iiv_tests = read_table("clinical_rescue_iiv/iiv_pfactor_age_interactions.csv")
    behavior_pf = read_table("clinical_rescue_behavior/ccd_behavior_pfactor_models.csv")
    mediation = read_table("clinical_rescue_behavior/ccd_behavior_mediation_models.csv")

    pf_tests = numeric(pf_tests, ["n", "p_value", "base_r2", "model_r2", "delta_r2", "q_value_global", "q_value_within_group"])
    pf_terms = numeric(pf_terms, ["estimate", "std_error", "t_value", "p_value", "q_value_global", "q_value_within_group"])
    iiv_tests = numeric(iiv_tests, ["n", "p_value", "main_r2", "interaction_r2", "delta_r2", "q_value"])
    behavior_pf = numeric(behavior_pf, ["n", "p_factor_estimate", "p_factor_se", "p_factor_p", "r2", "q_value"])
    mediation = numeric(
        mediation,
        [
            "n",
            "indirect_ab",
            "indirect_ci_low",
            "indirect_ci_high",
            "indirect_p_boot",
            "outcome_model_r2",
            "indirect_q_value",
        ],
    )

    pf_tests.to_csv(SOURCE_DIR / "clinical_pfactor_age_interaction_source.csv", index=False)
    pf_terms.to_csv(SOURCE_DIR / "clinical_pfactor_terms_source.csv", index=False)
    iiv_tests.to_csv(SOURCE_DIR / "clinical_iiv_interaction_source.csv", index=False)
    behavior_pf.to_csv(SOURCE_DIR / "clinical_behavior_pfactor_source.csv", index=False)
    mediation.to_csv(SOURCE_DIR / "clinical_behavior_mediation_source.csv", index=False)

    return {
        "pf_tests": pf_tests,
        "pf_terms": pf_terms,
        "iiv_tests": iiv_tests,
        "behavior_pf": behavior_pf,
        "mediation": mediation,
    }


def plot_clinical_figure(sources: dict[str, pd.DataFrame]) -> Path:
    pf_tests = sources["pf_tests"]
    pf_terms = sources["pf_terms"]
    iiv_tests = sources["iiv_tests"]
    behavior_pf = sources["behavior_pf"]
    mediation = sources["mediation"]

    fig = plt.figure(figsize=(11.2, 7.1))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.47, wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    trajectory = pf_tests[pf_tests["test"].eq("age_and_age2_x_pfactor_vs_base")]
    x = np.arange(len(FEATURES))
    width = 0.34
    for i, comp in enumerate(COMPONENTS):
        vals = []
        for feature in FEATURES:
            row = trajectory[trajectory["feature"].eq(feature) & trajectory["component"].eq(comp)]
            vals.append(float(row["delta_r2"].iloc[0]) * 100 if len(row) else np.nan)
        ax.bar(x + (i - 0.5) * width, vals, width=width, color=COMPONENT_COLORS[comp], label=COMPONENT_SHORT[comp])
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURES])
    ax.set_ylabel("Delta R2 (%)")
    ax.set_title("p-factor x age trajectory")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    heat_rows = []
    heat_labels = []
    for comp in COMPONENTS:
        row_vals = []
        for feature in FEATURES:
            row = trajectory[trajectory["feature"].eq(feature) & trajectory["component"].eq(comp)]
            row_vals.append(float(row["q_value_within_group"].iloc[0]) if len(row) else np.nan)
        heat_rows.append(minus_log10(np.asarray(row_vals)))
        heat_labels.append(COMPONENT_SHORT[comp])
    heat = np.asarray(heat_rows)
    image = ax.imshow(heat, aspect="auto", cmap="Blues", vmin=0, vmax=max(-np.log10(0.05), float(np.nanmax(heat))))
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURES])
    ax.set_yticks(np.arange(len(heat_labels)))
    ax.set_yticklabels(heat_labels)
    for yi in range(heat.shape[0]):
        for xi in range(heat.shape[1]):
            ax.text(xi, yi, f"{heat[yi, xi]:.2f}", ha="center", va="center", fontsize=6, color="#222222")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("-log10(q)", rotation=270, labelpad=9)
    cbar.ax.tick_params(labelsize=6)
    ax.set_title("Interaction FDR evidence")
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    iiv_order = [
        ("iiv_general", "General IIV", "#303030"),
        ("iiv_specific_sus_minus_ccd", "Specific IIV", "#6A5ACD"),
    ]
    for i, (outcome, label, color) in enumerate(iiv_order):
        vals = []
        for feature in FEATURES:
            row = iiv_tests[iiv_tests["feature"].eq(feature) & iiv_tests["outcome"].eq(outcome)]
            vals.append(float(row["delta_r2"].iloc[0]) * 100 if len(row) else np.nan)
        ax.bar(x + (i - 0.5) * width, vals, width=width, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURES])
    ax.set_ylabel("Delta R2 (%)")
    ax.set_title("Trial-to-trial variability")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 0])
    behavior_labels = {
        "ccd_accuracy": "Accuracy",
        "ccd_median_rt_correct": "Median RT",
        "ccd_rt_iiv_correct": "RT IIV",
    }
    behavior = behavior_pf.copy()
    behavior["label"] = behavior["mediator"].map(behavior_labels).fillna(behavior["mediator"])
    y = np.arange(len(behavior))[::-1]
    beta = behavior["p_factor_estimate"].to_numpy(dtype=float)
    se = behavior["p_factor_se"].to_numpy(dtype=float)
    ax.errorbar(beta, y, xerr=1.96 * se, fmt="o", color="#333333", ecolor="#888888", elinewidth=0.9, capsize=2)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(behavior["label"])
    ax.set_xlabel("p-factor coefficient")
    ax.set_title("p-factor to CCD behavior")
    for yi, q in zip(y, behavior["q_value"]):
        ax.text(ax.get_xlim()[1], yi, f"q={q:.2f}", ha="right", va="center", fontsize=6)
    panel_label(ax, "d")

    ax = fig.add_subplot(gs[1, 1])
    mediation_plot = mediation[
        mediation["feature"].isin(["n1", "p3"])
        & mediation["outcome"].isin(COMPONENTS)
        & mediation["mediator"].isin(["ccd_accuracy", "ccd_median_rt_correct", "ccd_rt_iiv_correct"])
    ].copy()
    mediation_plot["label"] = (
        mediation_plot["mediator"].map(behavior_labels).fillna(mediation_plot["mediator"])
        + " | "
        + mediation_plot["feature"].map(FEATURE_LABELS)
        + " "
        + mediation_plot["outcome"].map(COMPONENT_SHORT)
    )
    mediation_plot = mediation_plot.sort_values(["mediator", "feature", "outcome"])
    y = np.arange(len(mediation_plot))[::-1]
    point = mediation_plot["indirect_ab"].to_numpy(dtype=float)
    low = mediation_plot["indirect_ci_low"].to_numpy(dtype=float)
    high = mediation_plot["indirect_ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([point - low, high - point])
    ax.errorbar(point, y, xerr=xerr, fmt="o", markersize=2.8, color="#333333", ecolor="#8E8E8E", elinewidth=0.8, capsize=1.8)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(mediation_plot["label"], fontsize=5.6)
    ax.set_xlabel("Indirect effect")
    ax.set_title("Behavior mediation checks")
    panel_label(ax, "e")

    ax = fig.add_subplot(gs[1, 2])
    families = [
        (
            "ERP p x age",
            float(minus_log10(trajectory["q_value_within_group"]).max()),
            int(trajectory["n"].max()),
        ),
        (
            "IIV p x age",
            float(minus_log10(iiv_tests["q_value"]).max()),
            int(iiv_tests["n"].max()),
        ),
        (
            "p -> behavior",
            float(minus_log10(behavior_pf["q_value"]).max()),
            int(behavior_pf["n"].max()),
        ),
        (
            "Mediation",
            float(minus_log10(mediation["indirect_q_value"]).max()),
            int(mediation["n"].max()),
        ),
    ]
    fam = pd.DataFrame(families, columns=["family", "max_minus_log10_q", "n"])
    fam.to_csv(SOURCE_DIR / "clinical_rescue_family_summary.csv", index=False)
    y = np.arange(len(fam))[::-1]
    ax.barh(y, fam["max_minus_log10_q"], color=["#8A8A8A", "#8A8A8A", "#8A8A8A", "#8A8A8A"], height=0.58)
    ax.axvline(-np.log10(0.05), color="#C94F4F", linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(fam["family"])
    ax.set_xlabel("Maximum -log10(q)")
    ax.set_title("Rescue-analysis evidence ceiling")
    for yi, n in zip(y, fam["n"]):
        ax.text(ax.get_xlim()[1], yi, f"n={n:,}", ha="right", va="center", fontsize=6)
    panel_label(ax, "f")

    fig.suptitle("Psychopathology rescue analyses", y=0.99, fontsize=10, fontweight="bold")
    out = FIG_DIR / "figS3"
    save_pub(fig, out)
    plt.close(fig)
    return out


def write_report(outputs: list[Path], sample_sources: dict[str, pd.DataFrame]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    flow = sample_sources["sample_flow"]
    final_n = int(flow.loc[flow["stage"].eq("Full-channel selected ERP"), "subjects"].iloc[0])
    p_n = int(flow.loc[flow["stage"].eq("p-factor overlap"), "subjects"].iloc[0])
    lines = [
        "# Nature Companion Figures",
        "",
        "Generated with Python/matplotlib as publication-style companion figures.",
        "",
        "## Figure contract",
        "",
        "- Fig. 1 conclusion: the final selected-electrode/full-channel sample is large and QC is transparent enough to support the ERP analyses.",
        "- Fig. 2 conclusion: reliability is acceptable and the developmental effect is much stronger for task-general than task-specific ERP structure, especially P2/N450.",
        "- Fig. 3 conclusion: p-factor, IIV and CCD-behavior mediation checks do not rescue a strong clinical-dimension signal.",
        "",
        "## Key sample numbers",
        "",
        f"- Final full-channel selected ERP sample: n={final_n:,}.",
        f"- p-factor overlap used for interaction tests: n={p_n:,}.",
        "",
        "## Outputs",
        "",
    ]
    for out in outputs:
        lines.append(f"- `{out.relative_to(ROOT)}` (.svg/.pdf/.png/.tiff)")
    lines.extend(
        [
            "",
            "## Source data",
            "",
            f"- `{SOURCE_DIR.relative_to(ROOT)}`",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    sample_sources = prepare_sample_sources()
    development_sources = prepare_development_sources()
    clinical_sources = prepare_clinical_sources()

    outputs = [
        plot_sample_qc_figure(sample_sources),
        plot_development_figure(development_sources),
        plot_clinical_figure(clinical_sources),
    ]
    write_report(outputs, sample_sources)

    print("saved companion figures:")
    for out in outputs:
        print(out)
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

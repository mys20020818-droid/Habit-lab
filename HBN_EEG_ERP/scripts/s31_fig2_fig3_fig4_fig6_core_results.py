from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures" / "manuscript_core_nature"
SOURCE_DIR = TABLE_DIR / "manuscript_core_nature"
REPORT_PATH = ROOT / "results" / "reports" / "manuscript_core_nature_figures_report.md"

FEATURES = ["p1", "n1", "p3"]
FEATURE_LABELS = {"p1": "P1", "n1": "P2", "p3": "N450"}
AGE_GROUPS = ["child_5_9", "adolescent_10_14", "youth_15_21"]
AGE_LABELS = {
    "child_5_9": "5-9 y",
    "adolescent_10_14": "10-14 y",
    "youth_15_21": "15-21 y",
}
AGE_COLORS = {
    "child_5_9": "#8CBAD9",
    "adolescent_10_14": "#9BC48D",
    "youth_15_21": "#D7A067",
}
TASK_COLORS = {"CCD": "#6F8DBF", "SuS": "#4FA79D"}
GENERAL = "component_general"
SPECIFIC = "component_specific_sus_minus_ccd"
COMPONENT_LABELS = {GENERAL: "General", SPECIFIC: "Specific"}
COMPONENT_COLORS = {GENERAL: "#0F4D92", SPECIFIC: "#B64342"}
NEUTRAL_DARK = "#272727"
NEUTRAL_MID = "#767676"
NEUTRAL_LIGHT = "#D8D8D8"
NEUTRAL_FAINT = "#F2F2F2"
PREDICTORS = ["p_factor", "attention", "internalizing", "externalizing"]
PREDICTOR_LABELS = {
    "p_factor": "p-factor",
    "attention": "Attention",
    "internalizing": "Internalizing",
    "externalizing": "Externalizing",
}


def read_table(relative: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / relative)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "axes.linewidth": 0.65,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
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
        -0.09,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def format_q(q: float) -> str:
    if not np.isfinite(q):
        return "q=n/a"
    if q < 1e-3:
        return "q < 0.001"
    return f"q = {q:.3f}"


def star(q: float) -> str:
    if q < 1e-3:
        return "***"
    if q < 1e-2:
        return "**"
    if q < 0.05:
        return "*"
    return "n.s."


def fit_poly_ci(age: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None) -> dict[str, np.ndarray | float]:
    ok = np.isfinite(age) & np.isfinite(y)
    age = np.asarray(age[ok], dtype=float)
    y = np.asarray(y[ok], dtype=float)
    if grid is None:
        grid = np.linspace(5, 21, 161)
    center = float(np.mean(age))
    x = age - center
    xg = grid - center
    X = np.column_stack([np.ones_like(x), x, x**2])
    Xg = np.column_stack([np.ones_like(xg), xg, xg**2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    yhat = Xg @ beta
    resid = y - pred
    df = max(1, len(y) - X.shape[1])
    sigma2 = float(np.sum(resid**2) / df)
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(0, np.sum((Xg @ xtx_inv) * Xg, axis=1) * sigma2))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "grid": grid,
        "yhat": yhat,
        "lower": yhat - 1.96 * se,
        "upper": yhat + 1.96 * se,
        "r2": r2,
        "n": len(y),
    }


def binned_age_means(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df.copy()
    out["age_bin"] = out["age"].round().astype(int)
    return (
        out.groupby("age_bin", observed=False)
        .agg(
            n=("subject", "nunique"),
            mean=(value_col, "mean"),
            se=(value_col, lambda x: float(x.std(ddof=1) / np.sqrt(x.notna().sum()))),
        )
        .reset_index()
    )


def smooth_signal(y: np.ndarray, window: int = 13) -> np.ndarray:
    if window <= 1:
        return y
    window = int(window)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(np.asarray(y, dtype=float), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def load_components() -> pd.DataFrame:
    components = read_table("full_sus_ccd_components.csv")
    components = numeric(
        components,
        [
            "component_general",
            "component_specific_sus_minus_ccd",
            "general_odd",
            "general_even",
            "specific_odd",
            "specific_even",
            "ccd_z",
            "sus_z",
            "age",
            "p_factor",
            "attention",
            "internalizing",
            "externalizing",
        ],
    )
    components = components[components["age_group"].isin(AGE_GROUPS)].copy()
    return components


def prepare_figure2_sources() -> dict[str, pd.DataFrame]:
    variance = read_table("full_sus_ccd_variance_decomposition.csv")
    reliability = read_table("full_sus_ccd_component_reliability.csv")
    task_reliability = read_table("full_sus_ccd_split_half_reliability.csv")
    variance = numeric(variance, ["icc_general", "icc_specific"])
    reliability = numeric(reliability, ["spearman_brown", "pearson_r_odd_even"])
    task_reliability = numeric(task_reliability, ["spearman_brown", "pearson_r_odd_even"])
    variance["residual_or_other"] = np.maximum(0, 1 - variance["icc_general"] - variance["icc_specific"])
    variance.to_csv(SOURCE_DIR / "figure2_variance_decomposition.csv", index=False)
    reliability.to_csv(SOURCE_DIR / "figure2_component_reliability.csv", index=False)
    task_reliability.to_csv(SOURCE_DIR / "figure2_task_reliability_reference.csv", index=False)
    return {
        "variance": variance,
        "reliability": reliability,
        "task_reliability": task_reliability,
    }


def representative_subject() -> str:
    components = load_components()
    summary = read_table("selected_electrodes/selected_electrode_cached_recording_summary.csv")
    summary = numeric(summary, ["n_epochs_kept"])
    counts = summary.groupby(["subject", "task"], observed=False)["n_epochs_kept"].sum().unstack(fill_value=0)
    eligible = counts[(counts.get("CCD", 0) >= 60) & (counts.get("SuS", 0) >= 60)].index
    score_source = components[components["subject"].isin(eligible)].copy()
    score_source = score_source[score_source["feature"].isin(FEATURES)]
    spread = (
        score_source.groupby("subject", observed=False)
        .agg(
            age=("age", "first"),
            abs_score=(GENERAL, lambda x: float(np.mean(np.abs(x)))),
            abs_specific=(SPECIFIC, lambda x: float(np.mean(np.abs(x)))),
        )
        .reset_index()
    )
    spread = spread[spread["age"].between(9, 15, inclusive="both")].copy()
    if spread.empty:
        spread = score_source.groupby("subject", observed=False).agg(age=("age", "first"), abs_score=(GENERAL, lambda x: float(np.mean(np.abs(x))))).reset_index()
        spread["abs_specific"] = 0
    spread["representative_distance"] = (
        np.abs(spread["abs_score"] - spread["abs_score"].median())
        + 0.5 * np.abs(spread["abs_specific"] - spread["abs_specific"].median())
        + 0.03 * np.abs(spread["age"] - 12)
    )
    fallback = str(spread.sort_values("representative_distance").iloc[0]["subject"])
    for subject in spread.sort_values("representative_distance")["subject"].head(50):
        try:
            _, waves, _ = load_subject_task_waveforms(str(subject), roi="visual_posterior")
        except Exception:
            continue
        ccd = smooth_signal(waves["CCD"])
        sus = smooth_signal(waves["SuS"])
        general_wave = (ccd + sus) / 2
        peak = float(max(np.nanmax(np.abs(ccd)), np.nanmax(np.abs(sus))))
        robust_range = float(np.nanpercentile(general_wave, 97.5) - np.nanpercentile(general_wave, 2.5))
        if peak <= 9 and robust_range <= 6:
            return str(subject)
    return fallback


def load_subject_task_waveforms(subject: str, roi: str = "visual_posterior") -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    summary = read_table("selected_electrodes/selected_electrode_cached_recording_summary.csv")
    summary = numeric(summary, ["n_epochs_kept"])
    rois = read_table("selected_electrodes/selected_electrode_rois.csv")
    channels = rois[rois["roi"].eq(roi)]["channel"].tolist()
    subject_rows = summary[summary["subject"].eq(subject)].copy()
    waveforms: dict[str, np.ndarray] = {}
    weights: dict[str, int] = {}
    times: np.ndarray | None = None
    for task in ["CCD", "SuS"]:
        rows = subject_rows[subject_rows["task"].eq(task)]
        run_waves = []
        run_weights = []
        for _, row in rows.iterrows():
            cache_path = ROOT / Path(str(row["cache_path"]))
            if not cache_path.exists():
                continue
            with np.load(cache_path, allow_pickle=True) as z:
                evoked = z["evoked_uv"]
                ch_names = list(z["ch_names"].astype(str))
                times = z["times"].astype(float)
                picks = [ch_names.index(ch) for ch in channels if ch in ch_names]
                if not picks:
                    continue
                run_waves.append(np.nanmean(evoked[picks, :], axis=0))
                run_weights.append(float(row["n_epochs_kept"]))
        if run_waves:
            w = np.asarray(run_weights, dtype=float)
            arr = np.vstack(run_waves)
            waveforms[task] = np.average(arr, axis=0, weights=w)
            weights[task] = int(np.sum(w))
    if times is None or len(waveforms) < 2:
        raise RuntimeError(f"Could not load both task waveforms for {subject}")
    return times, waveforms, weights


def plot_fig2() -> Path:
    sources = prepare_figure2_sources()
    variance = sources["variance"].set_index("feature").reindex(FEATURES)
    reliability = sources["reliability"]
    task_reliability = sources["task_reliability"]

    fig = plt.figure(figsize=(11.4, 4.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.08, 1.36], wspace=0.44)
    x = np.arange(len(FEATURES))

    ax = fig.add_subplot(gs[0, 0])
    general = variance["icc_general"].to_numpy(dtype=float)
    specific = variance["icc_specific"].to_numpy(dtype=float)
    residual = variance["residual_or_other"].to_numpy(dtype=float)
    ax.bar(x, general, color=COMPONENT_COLORS[GENERAL], width=0.58, label="General")
    ax.bar(x, specific, bottom=general, color=COMPONENT_COLORS[SPECIFIC], width=0.58, label="Specific")
    ax.bar(x, residual, bottom=general + specific, color=NEUTRAL_LIGHT, width=0.58, label="Residual/other")
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURES])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Variance share")
    ax.set_title("Variance decomposition")
    ax.legend(fontsize=6, loc="upper right", handlelength=1.0)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    width = 0.28
    for i, comp in enumerate(["general", "specific_sus_minus_ccd"]):
        vals = []
        for feature in FEATURES:
            row = reliability[reliability["feature"].eq(feature) & reliability["component"].eq(comp)]
            vals.append(float(row["spearman_brown"].iloc[0]) if len(row) else np.nan)
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width=width,
            color=COMPONENT_COLORS[GENERAL] if comp == "general" else COMPONENT_COLORS[SPECIFIC],
            label="General" if comp == "general" else "Specific",
        )
    for task, marker, color in [("CCD", "o", TASK_COLORS["CCD"]), ("SuS", "s", TASK_COLORS["SuS"])]:
        vals = task_reliability[task_reliability["task"].eq(task)].set_index("feature").reindex(FEATURES)["spearman_brown"].to_numpy(dtype=float)
        ax.plot(x, vals, marker=marker, linestyle="None", color=color, markersize=4, label=f"{task} raw")
    ax.axhline(0.70, color=NEUTRAL_MID, linewidth=0.65, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURES])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Spearman-Brown")
    ax.set_title("Split-half reliability")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, handlelength=1.0, columnspacing=0.9)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    subject = representative_subject()
    times, waves, weights = load_subject_task_waveforms(subject, roi="visual_posterior")
    ccd = smooth_signal(waves["CCD"])
    sus = smooth_signal(waves["SuS"])
    general_wave = (ccd + sus) / 2
    ax.plot(times, ccd, color=TASK_COLORS["CCD"], linewidth=0.95, alpha=0.95, label=f"CCD, n={weights['CCD']}")
    ax.plot(times, sus, color=TASK_COLORS["SuS"], linewidth=0.95, alpha=0.95, label=f"SuS, n={weights['SuS']}")
    ax.plot(times, general_wave, color=COMPONENT_COLORS[GENERAL], linewidth=1.65, label="General mean")
    ax.fill_between(times, ccd, sus, color=COMPONENT_COLORS[SPECIFIC], alpha=0.13, linewidth=0, label="Task difference")
    ax.axvline(0, color=NEUTRAL_MID, linewidth=0.65)
    ax.axhline(0, color=NEUTRAL_LIGHT, linewidth=0.65)
    ax.axvspan(0.14, 0.22, color=COMPONENT_COLORS[GENERAL], alpha=0.08, linewidth=0)
    ax.axvspan(0.30, 0.60, color="#D99245", alpha=0.08, linewidth=0)
    ax.set_xlim(-0.05, 0.72)
    ymin = float(np.nanpercentile(np.r_[ccd, sus, general_wave], 1))
    ymax = float(np.nanpercentile(np.r_[ccd, sus, general_wave], 99))
    pad = max(0.75, 0.12 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Visual posterior ERP (uV)")
    ax.set_title("Single-subject decomposition")
    ax.legend(fontsize=6, loc="upper right", handlelength=1.2)
    panel_label(ax, "c")

    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.17, wspace=0.42)
    out = FIG_DIR / "fig2"
    save_pub(fig, out)
    plt.close(fig)
    return out


def prepare_trajectory_sources() -> dict[str, pd.DataFrame]:
    components = load_components()
    contrast = read_table("general_specific_development/general_specific_development_contrast_summary.csv")
    posthoc = read_table("full_sus_ccd_followup/full_sus_ccd_age_group_posthoc.csv")
    age_models = read_table("full_sus_ccd_age_models.csv")
    for df in [contrast, posthoc, age_models]:
        numeric(df, [c for c in df.columns if c not in {"feature", "component", "term", "contrast", "group_a", "group_b"}])
    components.to_csv(SOURCE_DIR / "figures3_4_component_scores.csv", index=False)
    contrast.to_csv(SOURCE_DIR / "figures3_4_development_contrast_summary.csv", index=False)
    posthoc.to_csv(SOURCE_DIR / "figure3_age_group_posthoc.csv", index=False)
    age_models.to_csv(SOURCE_DIR / "figures3_4_age_models.csv", index=False)
    return {"components": components, "contrast": contrast, "posthoc": posthoc, "age_models": age_models}


def add_pairwise_brackets(ax, posthoc: pd.DataFrame, feature: str) -> None:
    sub = posthoc[posthoc["feature"].eq(feature) & posthoc["component"].eq(GENERAL)].copy()
    group_pos = {g: i + 1 for i, g in enumerate(AGE_GROUPS)}
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    start = ymax - 0.18 * span
    step = 0.07 * span
    comparisons = [
        ("child_5_9", "adolescent_10_14"),
        ("adolescent_10_14", "youth_15_21"),
        ("child_5_9", "youth_15_21"),
    ]
    for i, (ga, gb) in enumerate(comparisons):
        row = sub[
            ((sub["group_a"].eq(ga)) & (sub["group_b"].eq(gb)))
            | ((sub["group_a"].eq(gb)) & (sub["group_b"].eq(ga)))
        ]
        if row.empty:
            continue
        q = float(row["q_value"].iloc[0])
        y = start + i * step
        x1, x2 = group_pos[ga], group_pos[gb]
        ax.plot([x1, x1, x2, x2], [y, y + 0.018 * span, y + 0.018 * span, y], color="#555555", linewidth=0.7)
        ax.text((x1 + x2) / 2, y + 0.021 * span, star(q), ha="center", va="bottom", fontsize=7)


def plot_scatter_fit(ax, components: pd.DataFrame, contrast: pd.DataFrame, feature: str) -> None:
    data = components[components["feature"].eq(feature)].copy()
    rng = np.random.default_rng(42)
    for age_group in AGE_GROUPS:
        sub = data[data["age_group"].eq(age_group)]
        x = sub["age"].to_numpy(dtype=float) + rng.normal(0, 0.025, len(sub))
        y = sub[GENERAL].to_numpy(dtype=float)
        ax.scatter(x, y, s=7, alpha=0.19, linewidth=0, color=AGE_COLORS[age_group], label=AGE_LABELS[age_group])
    fit = fit_poly_ci(data["age"].to_numpy(dtype=float), data[GENERAL].to_numpy(dtype=float))
    ax.plot(fit["grid"], fit["yhat"], color=NEUTRAL_DARK, linewidth=1.55)
    ax.fill_between(fit["grid"], fit["lower"], fit["upper"], color=NEUTRAL_DARK, alpha=0.11, linewidth=0)
    q = float(contrast.loc[contrast["feature"].eq(feature), "general_q_age_total"].iloc[0])
    ax.text(0.04, 0.94, f"n={int(fit['n']):,}\nR2={fit['r2'] * 100:.1f}%\n{format_q(q)}", transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.set_xlim(5, 21)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(f"{FEATURE_LABELS[feature]} general score (z)")
    ax.set_title(f"{FEATURE_LABELS[feature]} general developmental trajectory")


def plot_raincloud(ax, components: pd.DataFrame, posthoc: pd.DataFrame, feature: str) -> None:
    data = components[components["feature"].eq(feature)].copy()
    arrays = [data.loc[data["age_group"].eq(g), GENERAL].dropna().to_numpy(dtype=float) for g in AGE_GROUPS]
    violin = ax.violinplot(arrays, positions=np.arange(1, 4), widths=0.78, showextrema=False)
    for body, age_group in zip(violin["bodies"], AGE_GROUPS):
        body.set_facecolor(AGE_COLORS[age_group])
        body.set_edgecolor("none")
        body.set_alpha(0.28)
    bp = ax.boxplot(arrays, positions=np.arange(1, 4), widths=0.24, patch_artist=True, showfliers=False)
    for patch, age_group in zip(bp["boxes"], AGE_GROUPS):
        patch.set_facecolor(AGE_COLORS[age_group])
        patch.set_alpha(0.72)
        patch.set_edgecolor("#333333")
    for key in ["medians", "whiskers", "caps"]:
        for item in bp[key]:
            item.set_color("#333333")
            item.set_linewidth(0.75)
    rng = np.random.default_rng(123)
    for i, (arr, age_group) in enumerate(zip(arrays, AGE_GROUPS), start=1):
        if len(arr) > 350:
            arr = rng.choice(arr, size=350, replace=False)
        x = rng.normal(i + 0.29, 0.035, len(arr))
        ax.scatter(x, arr, s=5, alpha=0.20, color=AGE_COLORS[age_group], linewidth=0)
    ax.set_xticks(np.arange(1, 4))
    ax.set_xticklabels([AGE_LABELS[g] for g in AGE_GROUPS])
    ax.set_ylabel(f"{FEATURE_LABELS[feature]} general score (z)")
    ax.set_title(f"{FEATURE_LABELS[feature]} age-group distribution")
    add_pairwise_brackets(ax, posthoc, feature)


def plot_fig3(sources: dict[str, pd.DataFrame]) -> Path:
    components = sources["components"]
    contrast = sources["contrast"]
    posthoc = sources["posthoc"]

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.1), gridspec_kw={"hspace": 0.45, "wspace": 0.30})
    plot_scatter_fit(axes[0, 0], components, contrast, "n1")
    panel_label(axes[0, 0], "a")
    plot_scatter_fit(axes[0, 1], components, contrast, "p3")
    axes[0, 1].legend(fontsize=6, loc="lower right", handlelength=1.0)
    panel_label(axes[0, 1], "b")
    plot_raincloud(axes[1, 0], components, posthoc, "n1")
    panel_label(axes[1, 0], "c")
    plot_raincloud(axes[1, 1], components, posthoc, "p3")
    panel_label(axes[1, 1], "d")
    fig.subplots_adjust(top=0.93)
    out = FIG_DIR / "fig3"
    save_pub(fig, out)
    plt.close(fig)
    return out


def plot_component_curve(ax, components: pd.DataFrame, contrast: pd.DataFrame, feature: str) -> None:
    data = components[components["feature"].eq(feature)].copy()
    for comp, color in [(GENERAL, COMPONENT_COLORS[GENERAL]), (SPECIFIC, COMPONENT_COLORS[SPECIFIC])]:
        fit = fit_poly_ci(data["age"].to_numpy(dtype=float), data[comp].to_numpy(dtype=float))
        ax.plot(fit["grid"], fit["yhat"], color=color, linewidth=1.7, label=COMPONENT_LABELS[comp])
        ax.fill_between(fit["grid"], fit["lower"], fit["upper"], color=color, alpha=0.12, linewidth=0)
        binned = binned_age_means(data[["subject", "age", comp]].rename(columns={comp: "score"}), "score")
        ax.errorbar(binned["age_bin"], binned["mean"], yerr=binned["se"], fmt="o", markersize=2.6, linewidth=0.6, capsize=1.4, color=color, alpha=0.76)
    row = contrast[contrast["feature"].eq(feature)].iloc[0]
    ax.text(
        0.04,
        0.94,
        f"General/specific delta R2 ratio={row['general_to_specific_delta_r2_ratio']:.1f}x",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    ax.axhline(0, color="#D0D0D0", linewidth=0.7)
    ax.set_xlim(5, 21)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(f"{FEATURE_LABELS[feature]} score (z)")
    ax.set_title(f"{FEATURE_LABELS[feature]} general vs specific trajectories")


def plot_fig4(sources: dict[str, pd.DataFrame]) -> Path:
    components = sources["components"]
    contrast = sources["contrast"]
    reliability = read_table("full_sus_ccd_component_reliability.csv")
    variance = read_table("full_sus_ccd_variance_decomposition.csv")
    reliability = numeric(reliability, ["spearman_brown"])
    variance = numeric(variance, ["icc_general", "icc_specific"])
    contrast = numeric(contrast, ["general_delta_r2_age", "specific_delta_r2_age"])

    fig = plt.figure(figsize=(11.2, 6.9))
    gs = fig.add_gridspec(2, 2, hspace=0.43, wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    plot_component_curve(ax, components, contrast, "n1")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    plot_component_curve(ax, components, contrast, "p3")
    ax.legend(fontsize=6, loc="best", handlelength=1.3)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(FEATURES))
    width = 0.34
    general = contrast.set_index("feature").reindex(FEATURES)["general_delta_r2_age"].to_numpy(dtype=float) * 100
    specific = contrast.set_index("feature").reindex(FEATURES)["specific_delta_r2_age"].to_numpy(dtype=float) * 100
    ax.bar(x - width / 2, general, width=width, color=COMPONENT_COLORS[GENERAL], label="General")
    ax.bar(x + width / 2, specific, width=width, color=COMPONENT_COLORS[SPECIFIC], label="Specific")
    ratios = contrast.set_index("feature").reindex(FEATURES)["general_to_specific_delta_r2_ratio"].to_numpy(dtype=float)
    for xi, g, s, ratio in zip(x, general, specific, ratios):
        ax.text(xi, max(g, s) + 0.75, f"{ratio:.1f}x", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURES])
    ax.set_ylabel("Age delta R2 (%)")
    ax.set_title("Age effect size")
    ax.legend(fontsize=6, loc="upper left", handlelength=1.0)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    rows = []
    labels = []
    text_labels = []
    max_age_delta = float(np.nanmax([contrast["general_delta_r2_age"].max(), contrast["specific_delta_r2_age"].max()]))
    for feature in FEATURES:
        vrow = variance[variance["feature"].eq(feature)].iloc[0]
        rgen = reliability[reliability["feature"].eq(feature) & reliability["component"].eq("general")]["spearman_brown"].iloc[0]
        rspec = reliability[reliability["feature"].eq(feature) & reliability["component"].eq("specific_sus_minus_ccd")]["spearman_brown"].iloc[0]
        crow = contrast[contrast["feature"].eq(feature)].iloc[0]
        raw = [
            vrow["icc_general"],
            vrow["icc_specific"],
            rgen,
            rspec,
            crow["general_delta_r2_age"],
            crow["specific_delta_r2_age"],
        ]
        color_row = [
            raw[0],
            raw[1],
            raw[2],
            raw[3],
            raw[4] / max_age_delta if max_age_delta else np.nan,
            raw[5] / max_age_delta if max_age_delta else np.nan,
        ]
        rows.append(color_row)
        labels.append(FEATURE_LABELS[feature])
        text_labels.append([f"{raw[0]:.2f}", f"{raw[1]:.2f}", f"{raw[2]:.2f}", f"{raw[3]:.2f}", f"{raw[4] * 100:.1f}%", f"{raw[5] * 100:.1f}%"])
    matrix = np.asarray(rows, dtype=float)
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    columns = ["ICC\nGen", "ICC\nSpec", "Reliab.\nGen", "Reliab.\nSpec", "Age R2\nGen", "Age R2\nSpec"]
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns)
    for yi in range(matrix.shape[0]):
        for xi in range(matrix.shape[1]):
            ax.text(xi, yi, text_labels[yi][xi], ha="center", va="center", fontsize=6, color="#111111")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Metric-normalized strength", rotation=270, labelpad=11)
    cbar.ax.tick_params(labelsize=6)
    ax.set_title("Integrated evidence matrix")
    panel_label(ax, "d")

    fig.subplots_adjust(top=0.92)
    out = FIG_DIR / "fig4"
    save_pub(fig, out)
    plt.close(fig)
    return out


def plot_fig6(sources: dict[str, pd.DataFrame]) -> Path:
    components = sources["components"]
    psych = read_table("full_sus_ccd_psych_models.csv")
    pf_tests = read_table("pfactor_age_interaction/pfactor_age_interaction_model_tests.csv")
    psych = numeric(psych, ["estimate", "std_error", "p_value", "q_value", "n"])
    pf_tests = numeric(pf_tests, ["q_value_within_group"])
    psych.to_csv(SOURCE_DIR / "figure5_psychopathology_effects.csv", index=False)

    fig = plt.figure(figsize=(11.0, 4.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    col_pairs = [(feature, comp) for feature in FEATURES for comp in [GENERAL, SPECIFIC]]
    heat = np.zeros((len(PREDICTORS), len(col_pairs)))
    qvals = np.ones_like(heat)
    for yi, predictor in enumerate(PREDICTORS):
        for xi, (feature, comp) in enumerate(col_pairs):
            row = psych[psych["feature"].eq(feature) & psych["component"].eq(comp) & psych["predictor"].eq(predictor)]
            if row.empty:
                heat[yi, xi] = np.nan
                qvals[yi, xi] = np.nan
            else:
                heat[yi, xi] = float(row["estimate"].iloc[0])
                qvals[yi, xi] = float(row["q_value"].iloc[0])
    vmax = max(0.10, float(np.nanpercentile(np.abs(heat), 98)))
    image = ax.imshow(heat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(len(PREDICTORS)))
    ax.set_yticklabels([PREDICTOR_LABELS[p] for p in PREDICTORS])
    ax.set_xticks(np.arange(len(col_pairs)))
    ax.set_xticklabels([f"{FEATURE_LABELS[f]}\n{'Gen' if c == GENERAL else 'Spec'}" for f, c in col_pairs])
    for yi in range(heat.shape[0]):
        for xi in range(heat.shape[1]):
            mark = "n.s." if qvals[yi, xi] >= 0.05 else star(qvals[yi, xi])
            ax.text(xi, yi, f"{heat[yi, xi]:+.2f}\n{mark}", ha="center", va="center", fontsize=5.5, color="#111111")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Model coefficient", rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=6)
    ax.set_title("Psychopathology dimensions show near-zero ERP associations")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    data = components[components["feature"].eq("n1") & components["p_factor"].notna()].copy()
    data["pfactor_tertile"] = pd.qcut(data["p_factor"], 3, labels=["Low", "Middle", "High"])
    tertile_medians = data.groupby("pfactor_tertile", observed=False)["p_factor"].median().to_dict()
    tertile_colors = {"Low": "#6F8DBF", "Middle": "#8F8F8F", "High": "#B64342"}

    age = data["age"].to_numpy(dtype=float)
    y = data[GENERAL].to_numpy(dtype=float)
    p = data["p_factor"].to_numpy(dtype=float)
    ok = np.isfinite(age) & np.isfinite(y) & np.isfinite(p)
    age = age[ok]
    y = y[ok]
    p = p[ok]
    age_center = float(np.mean(age))
    x = age - age_center
    X = np.column_stack([np.ones_like(x), x, x**2, p])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = max(1, len(y) - X.shape[1])
    sigma2 = float(np.sum(resid**2) / df)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    grid = np.linspace(5, 21, 161)
    pred_rows = []
    for label in ["Low", "Middle", "High"]:
        p_value = float(tertile_medians[label])
        Xg = np.column_stack(
            [
                np.ones_like(grid),
                grid - age_center,
                (grid - age_center) ** 2,
                np.full_like(grid, p_value),
            ]
        )
        yhat = Xg @ beta
        se = np.sqrt(np.maximum(0, np.sum((Xg @ cov) * Xg, axis=1)))
        lower = yhat - 1.96 * se
        upper = yhat + 1.96 * se
        ax.plot(grid, yhat, color=tertile_colors[label], linewidth=1.5, label=f"{label} p-factor")
        ax.fill_between(grid, lower, upper, color=tertile_colors[label], alpha=0.10, linewidth=0)
        for xi, yi, lo, hi in zip(grid, yhat, lower, upper):
            pred_rows.append(
                {
                    "pfactor_tertile": label,
                    "pfactor_median": p_value,
                    "age": float(xi),
                    "predicted_n1_general": float(yi),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                }
            )
    pd.DataFrame(pred_rows).to_csv(SOURCE_DIR / "figure5_pfactor_tertile_no_interaction_predictions.csv", index=False)
    qrow = pf_tests[
        pf_tests["feature"].eq("n1")
        & pf_tests["component"].eq(GENERAL)
        & pf_tests["test"].eq("age_and_age2_x_pfactor_vs_base")
    ]
    q = float(qrow["q_value_within_group"].iloc[0]) if len(qrow) else np.nan
    ax.text(0.04, 0.94, f"p-factor x age interaction\n{format_q(q)}", transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.axhline(0, color="#D0D0D0", linewidth=0.7)
    ax.set_xlim(5, 21)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("P2 general score (z)")
    ax.set_title("Overlapping P2 developmental curves by p-factor tertile")
    ax.legend(fontsize=6, loc="lower left", handlelength=1.2)
    panel_label(ax, "b")

    fig.subplots_adjust(top=0.88)
    out = FIG_DIR / "fig6"
    save_pub(fig, out)
    plt.close(fig)
    return out


def write_report(outputs: list[Path]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manuscript Core Figures",
        "",
        "Generated with Python/matplotlib from the current HBN SuS+CCD analysis tables and full-channel ERP cache.",
        "",
        "## Figure set",
        "",
        "- fig1 is generated separately by `s30_fig1_study_design_flowchart.py`.",
        "- fig2: variance decomposition, component reliability, and single-subject SuS/CCD ERP decomposition.",
        "- fig3: P2/N450 task-general developmental hero figure.",
        "- fig4: general vs specific developmental separability.",
        "- fig6: psychopathology null-effect transparency figure.",
        "",
        "## Caution",
        "",
        "- Figure 2A residual/other is the remaining share after current general+specific decomposition, not a newly fitted LME residual variance component.",
        "- Figure 2B shows decomposed-component reliability plus raw task reliability markers; general/specific reliability is defined after task decomposition.",
        "",
        "## Outputs",
        "",
    ]
    for out in outputs:
        lines.append(f"- `{out.relative_to(ROOT)}` (.svg/.pdf/.png/.tiff)")
    lines.extend(["", "## Source data", "", f"- `{SOURCE_DIR.relative_to(ROOT)}`", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    trajectory_sources = prepare_trajectory_sources()
    outputs = [
        plot_fig2(),
        plot_fig3(trajectory_sources),
        plot_fig4(trajectory_sources),
        plot_fig6(trajectory_sources),
    ]
    write_report(outputs)
    print("saved manuscript figures:")
    for out in outputs:
        print(out)
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

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
COMPONENTS = ["component_general", "component_specific_sus_minus_ccd"]
MEDIATORS = ["ccd_accuracy", "ccd_median_rt_correct", "ccd_rt_iiv_correct"]


def events_path_from_raw(raw_path: str) -> Path:
    return Path(raw_path.replace("_eeg.set", "_events.tsv"))


def side_from_value(value: str) -> str | None:
    value = str(value)
    if value.startswith("left_"):
        return "left"
    if value.startswith("right_"):
        return "right"
    return None


def parse_ccd_events(row: pd.Series) -> list[dict]:
    events_path = events_path_from_raw(str(row["raw_path"]))
    if not events_path.exists():
        return []
    try:
        events = pd.read_csv(events_path, sep="\t")
    except Exception:
        return []
    if "onset" not in events.columns or "value" not in events.columns:
        return []
    events = events.copy()
    events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
    events = events.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)
    values = events["value"].astype(str)
    target_idx = events.index[values.isin(["left_target", "right_target"])].to_list()
    button_mask = values.isin(["left_buttonPress", "right_buttonPress"])
    rows: list[dict] = []
    for pos, idx in enumerate(target_idx):
        target = events.loc[idx]
        next_target_onset = (
            float(events.loc[target_idx[pos + 1], "onset"]) if pos + 1 < len(target_idx) else np.inf
        )
        target_side = side_from_value(target["value"])
        candidates = events[(events.index > idx) & button_mask & (events["onset"] < next_target_onset)]
        response = candidates.iloc[0] if not candidates.empty else None
        if response is None:
            response_side = None
            rt = np.nan
            feedback = ""
            correct = False
            responded = False
        else:
            response_side = side_from_value(response["value"])
            rt = float(response["onset"] - target["onset"])
            feedback = str(response.get("feedback", ""))
            correct = bool((response_side == target_side) and ("smiley_face" in feedback))
            responded = True
        rows.append(
            {
                "subject": row["subject"],
                "release": row.get("release", ""),
                "run": row.get("run", ""),
                "raw_path": row["raw_path"],
                "events_path": str(events_path),
                "target_onset": float(target["onset"]),
                "target_side": target_side,
                "responded": responded,
                "response_side": response_side,
                "rt": rt,
                "feedback": feedback,
                "correct": correct,
            }
        )
    return rows


def build_behavior(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ccd = manifest[manifest["task"].eq("CCD")].copy()
    trial_rows: list[dict] = []
    for _, row in ccd.iterrows():
        trial_rows.extend(parse_ccd_events(row))
    trials = pd.DataFrame(trial_rows)
    if trials.empty:
        return trials, pd.DataFrame()
    trials["rt"] = pd.to_numeric(trials["rt"], errors="coerce")
    trials["correct"] = trials["correct"].astype(bool)
    trials["responded"] = trials["responded"].astype(bool)

    rows: list[dict] = []
    for subject, sub in trials.groupby("subject"):
        correct_rt = sub.loc[sub["correct"], "rt"].dropna()
        response_rt = sub.loc[sub["responded"], "rt"].dropna()
        n_targets = len(sub)
        n_correct = int(sub["correct"].sum())
        rows.append(
            {
                "subject": subject,
                "ccd_n_targets": n_targets,
                "ccd_n_correct": n_correct,
                "ccd_n_responded": int(sub["responded"].sum()),
                "ccd_accuracy": n_correct / n_targets if n_targets else np.nan,
                "ccd_response_rate": float(sub["responded"].mean()) if n_targets else np.nan,
                "ccd_median_rt_correct": float(correct_rt.median()) if len(correct_rt) else np.nan,
                "ccd_mean_rt_correct": float(correct_rt.mean()) if len(correct_rt) else np.nan,
                "ccd_rt_iiv_correct": float(correct_rt.std(ddof=1)) if len(correct_rt) > 1 else np.nan,
                "ccd_median_rt_all_responses": float(response_rt.median()) if len(response_rt) else np.nan,
            }
        )
    return trials, pd.DataFrame(rows)


def prep_model_df(components: pd.DataFrame, behavior: pd.DataFrame, mediator: str, outcome: str, feature: str) -> pd.DataFrame:
    df = components[components["feature"].eq(feature)].merge(behavior, on="subject", how="inner")
    for col in ["age", "p_factor", mediator, outcome]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[outcome, mediator, "age", "sex", "p_factor"]).copy()
    if mediator != "ccd_accuracy":
        df = df[df[mediator] > 0].copy()
        df[f"log_{mediator}"] = np.log(df[mediator])
        med_col = f"log_{mediator}"
    else:
        med_col = mediator
    df["age_c"] = df["age"] - df["age"].mean()
    df["age_c2"] = df["age_c"] ** 2
    df["p_c"] = df["p_factor"] - df["p_factor"].mean()
    df["m_z"] = (df[med_col] - df[med_col].mean()) / df[med_col].std(ddof=1)
    return df


def fit_behavior_models(behavior: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    df = behavior.merge(metadata.drop_duplicates("subject"), on="subject", how="left")
    rows: list[dict] = []
    for mediator in MEDIATORS:
        model_df = df.copy()
        for col in ["age", "p_factor", mediator]:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
        model_df = model_df.dropna(subset=[mediator, "age", "sex", "p_factor"]).copy()
        if mediator != "ccd_accuracy":
            model_df = model_df[model_df[mediator] > 0].copy()
            outcome = f"log_{mediator}"
            model_df[outcome] = np.log(model_df[mediator])
        else:
            outcome = mediator
        model_df["age_c"] = model_df["age"] - model_df["age"].mean()
        model_df["age_c2"] = model_df["age_c"] ** 2
        model_df["p_c"] = model_df["p_factor"] - model_df["p_factor"].mean()
        if len(model_df) < 30:
            continue
        model = smf.ols(f"{outcome} ~ age_c + age_c2 + C(sex) + p_c", data=model_df).fit()
        rows.append(
            {
                "mediator": mediator,
                "outcome_modeled": outcome,
                "n": int(model.nobs),
                "p_factor_estimate": float(model.params.get("p_c", np.nan)),
                "p_factor_se": float(model.bse.get("p_c", np.nan)),
                "p_factor_t": float(model.tvalues.get("p_c", np.nan)),
                "p_factor_p": float(model.pvalues.get("p_c", np.nan)),
                "r2": float(model.rsquared),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = multipletests(out["p_factor_p"].fillna(1.0), method="fdr_bh")[1]
    return out


def bootstrap_indirect(df: pd.DataFrame, outcome: str, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = []
    n = len(df)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = df.iloc[idx].copy()
        try:
            a = smf.ols("m_z ~ age_c + age_c2 + C(sex) + p_c", data=sample).fit().params.get("p_c", np.nan)
            b = smf.ols(f"{outcome} ~ age_c + age_c2 + C(sex) + p_c + m_z", data=sample).fit().params.get("m_z", np.nan)
        except Exception:
            continue
        if np.isfinite(a) and np.isfinite(b):
            values.append(float(a * b))
    if not values:
        return np.nan, np.nan, np.nan
    arr = np.asarray(values)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    p = 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))
    return float(lo), float(hi), float(min(p, 1.0))


def fit_mediation(components: pd.DataFrame, behavior: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows: list[dict] = []
    for mediator in MEDIATORS:
        for feature in FEATURES:
            for outcome in COMPONENTS:
                df = prep_model_df(components, behavior, mediator, outcome, feature)
                if len(df) < 50 or df["m_z"].std(ddof=1) == 0:
                    continue
                a_model = smf.ols("m_z ~ age_c + age_c2 + C(sex) + p_c", data=df).fit()
                b_model = smf.ols(f"{outcome} ~ age_c + age_c2 + C(sex) + p_c + m_z", data=df).fit()
                total_model = smf.ols(f"{outcome} ~ age_c + age_c2 + C(sex) + p_c", data=df).fit()
                a = float(a_model.params.get("p_c", np.nan))
                b = float(b_model.params.get("m_z", np.nan))
                indirect = a * b
                ci_low, ci_high, p_boot = bootstrap_indirect(
                    df,
                    outcome,
                    n_boot=n_boot,
                    seed=20260511 + len(rows),
                )
                rows.append(
                    {
                        "mediator": mediator,
                        "feature": feature,
                        "outcome": outcome,
                        "n": len(df),
                        "a_pfactor_to_mediator": a,
                        "a_p_value": float(a_model.pvalues.get("p_c", np.nan)),
                        "b_mediator_to_erp": b,
                        "b_p_value": float(b_model.pvalues.get("m_z", np.nan)),
                        "direct_pfactor_to_erp": float(b_model.params.get("p_c", np.nan)),
                        "direct_p_value": float(b_model.pvalues.get("p_c", np.nan)),
                        "total_pfactor_to_erp": float(total_model.params.get("p_c", np.nan)),
                        "total_p_value": float(total_model.pvalues.get("p_c", np.nan)),
                        "indirect_ab": float(indirect),
                        "indirect_ci_low": ci_low,
                        "indirect_ci_high": ci_high,
                        "indirect_p_boot": p_boot,
                        "outcome_model_r2": float(b_model.rsquared),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["indirect_q_value"] = multipletests(out["indirect_p_boot"].fillna(1.0), method="fdr_bh")[1]
    return out


def plot_behavior_mediation(behavior_models: pd.DataFrame, mediation: pd.DataFrame, out_base: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), gridspec_kw={"width_ratios": [0.85, 1.55]})

    ax = axes[0]
    bm = behavior_models.copy()
    y = np.arange(len(bm))
    ax.axvline(0, color="#D8D8D8", linewidth=0.9)
    ax.scatter(bm["p_factor_estimate"], y, color="#0F4D92", s=38)
    for yi, (_, row) in zip(y, bm.iterrows()):
        ax.text(row["p_factor_estimate"], yi + 0.18, f"q={row['q_value']:.3g}", ha="center", va="bottom", fontsize=6)
    ax.set_yticks(y, bm["mediator"].str.replace("ccd_", "", regex=False))
    ax.set_xlabel("p-factor estimate")
    ax.set_title("p-factor to CCD behavior", fontsize=8)

    ax = axes[1]
    top = mediation.sort_values("indirect_p_boot").head(12).copy()
    y = np.arange(len(top))
    colors = ["#B64342" if q < 0.05 else "#858585" for q in top["indirect_q_value"]]
    labels = [
        f"{row.feature.upper()} {'Gen' if row.outcome == 'component_general' else 'Spec'} via {row.mediator.replace('ccd_', '')}"
        for row in top.itertuples()
    ]
    ax.axvline(0, color="#D8D8D8", linewidth=0.9)
    ax.scatter(top["indirect_ab"], y, color=colors, s=34)
    for yi, (_, row) in zip(y, top.iterrows()):
        ax.plot([row["indirect_ci_low"], row["indirect_ci_high"]], [yi, yi], color="#858585", linewidth=0.8)
    ax.set_yticks(y, labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Indirect effect a*b")
    ax.set_title("Smallest bootstrap indirect paths", fontsize=8)

    fig.suptitle("CCD behavior mediation screen", y=1.02, fontsize=9, fontweight="bold")
    fig.tight_layout(w_pad=1.5)
    ensure_parent(str(out_base))
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--components", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    components = pd.read_csv(args.components)
    metadata = pd.read_csv(args.metadata)

    trials, behavior = build_behavior(manifest)
    behavior_models = fit_behavior_models(behavior, metadata)
    mediation = fit_mediation(components, behavior, n_boot=args.n_boot)

    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    trials.to_csv(out_dir / "ccd_behavior_trials.csv", index=False)
    behavior.to_csv(out_dir / "ccd_behavior_by_subject.csv", index=False)
    behavior_models.to_csv(out_dir / "ccd_behavior_pfactor_models.csv", index=False)
    mediation.to_csv(out_dir / "ccd_behavior_mediation_models.csv", index=False)
    plot_behavior_mediation(behavior_models, mediation, figures_dir / "ccd_behavior_mediation_screen")

    summary = pd.DataFrame(
        [
            {"table": "ccd_behavior_trials", "rows": len(trials), "subjects": trials["subject"].nunique() if not trials.empty else 0},
            {"table": "ccd_behavior_by_subject", "rows": len(behavior), "subjects": behavior["subject"].nunique() if not behavior.empty else 0},
            {"table": "ccd_behavior_mediation_models", "rows": len(mediation), "subjects": np.nan},
        ]
    )
    summary.to_csv(out_dir / "ccd_behavior_sample_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(behavior_models.to_string(index=False))
    print(mediation.sort_values("indirect_p_boot").head(12).to_string(index=False))


if __name__ == "__main__":
    main()

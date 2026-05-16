from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from s00_utils_common import ensure_parent, load_config
from s20_model_component_decomposition import build_components, component_reliability, variance_decomposition


AGE_GROUPS = ["child_5_9", "adolescent_10_14", "youth_15_21"]
TASKS = ["CCD", "SuS"]
FEATURES = ["p1", "n1", "p3"]
FRONTAL_SURROGATE = ["E1", "E8", "E14", "E21", "E25", "E32", "E125", "E126", "E127", "E128"]


def raw_stem(path: Path) -> str:
    for suffix in ["_eeg.set", "_eeg.fif", "_eeg.edf", "_eeg.bdf", "_eeg.mff"]:
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def events_path_for_raw(raw_path: Path) -> Path:
    return raw_path.with_name(f"{raw_stem(raw_path)}_events.tsv")


def visual_event_mask(task: str, values: pd.Series) -> pd.Series:
    values = values.fillna("").astype(str)
    if task == "SuS":
        return values.eq("stim_ON")
    if task == "CCD":
        return values.eq("contrastTrial_start")
    return values.str.contains("stim_ON|contrastTrial_start", regex=True)


def make_events_array(events: pd.DataFrame, raw_sfreq: float) -> np.ndarray:
    if "sample" in events.columns:
        samples = pd.to_numeric(events["sample"], errors="coerce")
    else:
        samples = pd.to_numeric(events["onset"], errors="coerce") * raw_sfreq
    samples = samples.dropna().astype(int)
    return np.column_stack([samples.to_numpy(), np.zeros(len(samples), dtype=int), np.ones(len(samples), dtype=int)])


def configured_channels(config: dict) -> dict[str, list[str]]:
    ch = config.get("features", {}).get("channels", {})
    return {name: list(vals) for name, vals in ch.items()}


def configured_analysis_picks(config: dict, ch_names: list[str]) -> list[int]:
    channels = configured_channels(config)
    wanted: list[str] = []
    for spec in config.get("features", {}).get("windows", {}).values():
        wanted.extend(channels.get(spec.get("channel_group", ""), []))
    wanted_set = set(wanted)
    return [idx for idx, ch in enumerate(ch_names) if ch in wanted_set]


def select_subjects(
    components: pd.DataFrame,
    qc: pd.DataFrame,
    n_per_group: int,
    random_state: int,
    min_epochs_per_task: int,
    max_drop_fraction: float,
) -> pd.DataFrame:
    meta = components[["subject", "age", "age_group", "sex"]].drop_duplicates("subject")
    meta = meta[meta["age_group"].isin(AGE_GROUPS)].copy()
    ok = qc[qc["status"].eq("ok") & qc["task"].isin(TASKS)].copy()
    ok["n_epochs_kept"] = pd.to_numeric(ok["n_epochs_kept"], errors="coerce")
    ok["drop_fraction"] = pd.to_numeric(ok["drop_fraction"], errors="coerce")
    task_quality = ok.groupby(["subject", "task"], as_index=False).agg(
        n_epochs_kept=("n_epochs_kept", "sum"),
        drop_fraction=("drop_fraction", "mean"),
        n_recordings=("raw_path", "nunique"),
    )
    wide = task_quality.pivot(index="subject", columns="task", values="n_epochs_kept")
    drop = task_quality.pivot(index="subject", columns="task", values="drop_fraction")
    eligible = wide.dropna(subset=TASKS)
    eligible = eligible[(eligible[TASKS] >= min_epochs_per_task).all(axis=1)]
    drop = drop.reindex(eligible.index)
    eligible = eligible[(drop[TASKS] <= max_drop_fraction).all(axis=1)]
    meta = meta[meta["subject"].isin(eligible.index)].copy()
    return (
        meta.groupby("age_group", group_keys=False)
        .apply(lambda d: d.sample(n=min(n_per_group, len(d)), random_state=random_state))
        .sort_values(["age_group", "subject"])
        .reset_index(drop=True)
    )


def fit_ica_clean_epochs(raw_path: Path, task: str, config: dict, args: argparse.Namespace) -> tuple[object, dict]:
    import mne
    from mne.preprocessing import ICA

    events_path = events_path_for_raw(raw_path)
    events_tsv = pd.read_csv(events_path, sep="\t")
    visual_events = events_tsv[visual_event_mask(task, events_tsv["value"])].copy()
    if visual_events.empty:
        raise ValueError("No visual/trial-onset events found")

    raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose="ERROR")
    raw.pick("eeg")
    raw.notch_filter(freqs=[args.notch_freq], verbose="ERROR")
    raw.filter(l_freq=args.l_freq, h_freq=args.h_freq, verbose="ERROR")
    raw.set_eeg_reference(args.reference, verbose="ERROR")

    fit_raw = raw.copy()
    if args.resample_sfreq:
        fit_raw.resample(args.resample_sfreq, npad="auto", verbose="ERROR")

    n_components = min(args.n_components, max(2, len(fit_raw.ch_names) - 1))
    ica = ICA(n_components=n_components, method=args.ica_method, random_state=args.random_state, max_iter=args.max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica.fit(fit_raw, decim=args.decim, verbose="ERROR")

    sources = ica.get_sources(raw).get_data()
    frontal = [ch for ch in FRONTAL_SURROGATE if ch in raw.ch_names]
    exclude: list[int] = []
    corr_scores = []
    if frontal:
        frontal_signal = raw.copy().pick(frontal).get_data().mean(axis=0)
        n = min(sources.shape[1], frontal_signal.shape[0])
        for idx in range(sources.shape[0]):
            if n < 10:
                corr = 0.0
            else:
                corr = pearsonr(sources[idx, :n], frontal_signal[:n]).statistic
            corr_scores.append(float(corr))
            if abs(corr) >= args.frontal_corr_threshold:
                exclude.append(idx)
    # Guard against over-correction in this sensitivity analysis.
    exclude = exclude[: args.max_excluded_components]
    ica.exclude = exclude
    cleaned = raw.copy()
    ica.apply(cleaned, verbose="ERROR")

    events_array = make_events_array(visual_events, float(cleaned.info["sfreq"]))
    epochs = mne.Epochs(
        cleaned,
        events_array,
        event_id={"visual_onset": 1},
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=(args.baseline_tmin, args.baseline_tmax),
        preload=True,
        reject=None,
        detrend=0,
        verbose="ERROR",
    )
    p2p_uv = np.array([])
    if args.reject_uv > 0 and len(epochs):
        data = epochs.get_data(copy=False)
        reject_picks = configured_analysis_picks(config, epochs.ch_names)
        if reject_picks:
            p2p_uv = np.ptp(data[:, reject_picks, :], axis=2).max(axis=1) * 1e6
            bad_idx = np.where(p2p_uv > args.reject_uv)[0]
            if len(bad_idx):
                epochs.drop(bad_idx, reason="analysis_channel_amplitude", verbose="ERROR")
    qc = {
        "n_visual_events": len(events_array),
        "n_epochs_kept": len(epochs),
        "n_channels": len(raw.ch_names),
        "n_ica_components": int(n_components),
        "n_components_excluded": len(exclude),
        "excluded_components": ",".join(map(str, exclude)),
        "max_abs_frontal_corr": float(np.max(np.abs(corr_scores))) if corr_scores else np.nan,
        "p2p_uv_median_analysis_channels": float(np.median(p2p_uv)) if len(p2p_uv) else np.nan,
        "p2p_uv_p90_analysis_channels": float(np.quantile(p2p_uv, 0.90)) if len(p2p_uv) else np.nan,
    }
    return epochs, qc


def extract_features_from_epochs(epochs, config: dict) -> list[dict]:
    channels = configured_channels(config)
    rows = []
    if len(epochs) == 0:
        return rows
    data = epochs.get_data(copy=False)
    times = epochs.times
    for feature, spec in config["features"]["windows"].items():
        if feature not in FEATURES:
            continue
        group = spec["channel_group"]
        picks = [epochs.ch_names.index(ch) for ch in channels.get(group, []) if ch in epochs.ch_names]
        if not picks:
            continue
        mask = (times >= float(spec["tmin"])) & (times <= float(spec["tmax"]))
        values = data[:, picks][:, :, mask].mean(axis=(1, 2)) * 1e6
        odd = values[::2]
        even = values[1::2]
        rows.append(
            {
                "feature": feature,
                "channel_group": group,
                "n_recordings": 1,
                "n_epochs": int(len(values)),
                "amplitude_uv": float(np.mean(values)) if len(values) else np.nan,
                "split_odd_uv": float(np.mean(odd)) if len(odd) else np.nan,
                "split_even_uv": float(np.mean(even)) if len(even) else np.nan,
            }
        )
    return rows


def aggregate_subject_task(features: pd.DataFrame) -> pd.DataFrame:
    weighted = features.copy()
    for col in ["amplitude_uv", "split_odd_uv", "split_even_uv"]:
        weighted[f"weighted_{col}"] = weighted[col] * weighted["n_epochs"]
    grouped = (
        weighted.groupby(["subject", "task", "feature", "channel_group"], as_index=False)
        .agg(
            n_recordings=("raw_path", "nunique"),
            n_epochs=("n_epochs", "sum"),
            amplitude_uv=("weighted_amplitude_uv", "sum"),
            split_odd_uv=("weighted_split_odd_uv", "sum"),
            split_even_uv=("weighted_split_even_uv", "sum"),
        )
    )
    grouped["amplitude_uv"] = grouped["amplitude_uv"] / grouped["n_epochs"]
    grouped["split_odd_uv"] = grouped["split_odd_uv"] / grouped["n_epochs"]
    grouped["split_even_uv"] = grouped["split_even_uv"] / grouped["n_epochs"]
    return grouped


def has_dual_task_data(subject_task: pd.DataFrame) -> bool:
    if subject_task.empty:
        return False
    for feature, df in subject_task.groupby("feature"):
        wide = df.pivot(index="subject", columns="task", values="amplitude_uv")
        if set(TASKS).issubset(set(wide.columns)) and wide.dropna(subset=TASKS).shape[0] > 0:
            return True
    return False


def compare_with_primary(ica_components: pd.DataFrame, primary_components: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        left = ica_components[ica_components["feature"].eq(feature)].set_index("subject")
        right = primary_components[primary_components["feature"].eq(feature)].set_index("subject")
        common = left.index.intersection(right.index)
        for component in ["component_general", "component_specific_sus_minus_ccd"]:
            x = left.loc[common, component]
            y = right.loc[common, component]
            valid = x.notna() & y.notna()
            if valid.sum() >= 5:
                r = pearsonr(x[valid], y[valid]).statistic
                mad = np.mean(np.abs(x[valid] - y[valid]))
            else:
                r = np.nan
                mad = np.nan
            rows.append(
                {
                    "feature": feature,
                    "component": component,
                    "n_subjects": int(valid.sum()),
                    "primary_ica_pearson_r": float(r) if pd.notna(r) else np.nan,
                    "mean_absolute_score_difference": float(mad) if pd.notna(mad) else np.nan,
                    "primary_mean": float(y[valid].mean()) if valid.sum() else np.nan,
                    "ica_mean": float(x[valid].mean()) if valid.sum() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--manifest", default="results/tables/full_sus_ccd_decomp_manifest_clean.csv")
    parser.add_argument("--qc", default="results/tables/full_sus_ccd_epoch_qc_analysisch_reject250.csv")
    parser.add_argument("--components", default="results/tables/full_sus_ccd_components.csv")
    parser.add_argument("--out-dir", default="results/tables/ica_sensitivity")
    parser.add_argument("--n-per-group", type=int, default=6)
    parser.add_argument("--min-epochs-per-task", type=int, default=60)
    parser.add_argument("--max-drop-fraction", type=float, default=0.25)
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--notch-freq", type=float, default=60.0)
    parser.add_argument("--reference", default="average")
    parser.add_argument("--reject-uv", type=float, default=250.0)
    parser.add_argument("--tmin", type=float, default=-0.2)
    parser.add_argument("--tmax", type=float, default=0.8)
    parser.add_argument("--baseline-tmin", type=float, default=-0.2)
    parser.add_argument("--baseline-tmax", type=float, default=0.0)
    parser.add_argument("--resample-sfreq", type=float, default=250.0)
    parser.add_argument("--ica-method", default="fastica")
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--max-iter", default="auto")
    parser.add_argument("--decim", type=int, default=5)
    parser.add_argument("--frontal-corr-threshold", type=float, default=0.30)
    parser.add_argument("--max-excluded-components", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=20260513)
    parser.add_argument("--max-recordings", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = pd.read_csv(args.manifest)
    qc = pd.read_csv(args.qc)
    primary = pd.read_csv(args.components)
    selected = select_subjects(
        primary,
        qc,
        args.n_per_group,
        args.random_state,
        args.min_epochs_per_task,
        args.max_drop_fraction,
    )
    rows_manifest = manifest[manifest["subject"].isin(selected["subject"]) & manifest["task"].isin(TASKS)].copy()
    if args.max_recordings > 0:
        rows_manifest = rows_manifest.head(args.max_recordings).copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_dir / "ica_sensitivity_selected_subjects.csv", index=False)
    rows_manifest.to_csv(out_dir / "ica_sensitivity_manifest.csv", index=False)

    feature_rows = []
    qc_rows = []
    for _, row in rows_manifest.iterrows():
        raw_path = Path(row["raw_path"])
        base = {
            "subject": row["subject"],
            "task": row["task"],
            "raw_task": row.get("raw_task", ""),
            "run": row.get("run", ""),
            "raw_path": str(raw_path),
        }
        try:
            epochs, one_qc = fit_ica_clean_epochs(raw_path, str(row["task"]), config, args)
            for feat in extract_features_from_epochs(epochs, config):
                feature_rows.append({**base, **feat})
            qc_rows.append({**base, "status": "ok", **one_qc, "error": ""})
        except Exception as exc:
            qc_rows.append({**base, "status": "failed", "error": str(exc)})

    features = pd.DataFrame(feature_rows)
    qc_out = pd.DataFrame(qc_rows)
    features.to_csv(out_dir / "ica_sensitivity_features_by_recording.csv", index=False)
    qc_out.to_csv(out_dir / "ica_sensitivity_qc.csv", index=False)
    if features.empty:
        raise RuntimeError("No ICA features were extracted")

    subject_task = aggregate_subject_task(features)
    subject_task.to_csv(out_dir / "ica_sensitivity_features_by_subject_task.csv", index=False)
    if not has_dual_task_data(subject_task):
        summary = pd.DataFrame(
            [
                {
                    "n_selected_subjects": selected["subject"].nunique(),
                    "n_successful_recordings": int(qc_out["status"].eq("ok").sum()),
                    "n_failed_recordings": int(qc_out["status"].eq("failed").sum()),
                    "n_component_subjects": 0,
                    "mean_excluded_components": float(pd.to_numeric(qc_out.get("n_components_excluded"), errors="coerce").mean()),
                    "median_epochs_kept": float(pd.to_numeric(qc_out.get("n_epochs_kept"), errors="coerce").median()),
                    "status": "no_dual_task_subjects_after_ica_epoching",
                }
            ]
        )
        summary.to_csv(out_dir / "ica_sensitivity_summary.csv", index=False)
        print(summary.to_string(index=False))
        return
    ica_components = build_components(subject_task, primary.drop_duplicates("subject"))
    ica_components.to_csv(out_dir / "ica_sensitivity_components.csv", index=False)
    variance = variance_decomposition(subject_task)
    reliability = component_reliability(ica_components)
    comparison = compare_with_primary(ica_components, primary)
    variance.to_csv(out_dir / "ica_sensitivity_variance_decomposition.csv", index=False)
    reliability.to_csv(out_dir / "ica_sensitivity_reliability.csv", index=False)
    comparison.to_csv(out_dir / "ica_sensitivity_primary_comparison.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "n_selected_subjects": selected["subject"].nunique(),
                "n_successful_recordings": int(qc_out["status"].eq("ok").sum()),
                "n_failed_recordings": int(qc_out["status"].eq("failed").sum()),
                "n_component_subjects": int(ica_components["subject"].nunique()),
                "mean_excluded_components": float(pd.to_numeric(qc_out.get("n_components_excluded"), errors="coerce").mean()),
                "median_epochs_kept": float(pd.to_numeric(qc_out.get("n_epochs_kept"), errors="coerce").median()),
            }
        ]
    )
    summary.to_csv(out_dir / "ica_sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

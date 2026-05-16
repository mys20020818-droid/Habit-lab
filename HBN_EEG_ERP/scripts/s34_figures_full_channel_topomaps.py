from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from s00_utils_common import ensure_parent, load_config
from s10_preprocess_visual_epochs import events_path_for_raw, make_events_array, visual_event_mask


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7


AGE_GROUPS = ["child_5_9", "adolescent_10_14", "youth_15_21"]
AGE_LABELS = {"child_5_9": "5-9", "adolescent_10_14": "10-14", "youth_15_21": "15-21"}
AGE_COLORS = {"child_5_9": "#8FB8DE", "adolescent_10_14": "#D9A441", "youth_15_21": "#B64342"}
COMPONENTS = ["general", "specific_sus_minus_ccd"]
COMPONENT_LABELS = {"general": "Task-general", "specific_sus_minus_ccd": "Task-specific SuS-CCD"}
FEATURE_WINDOWS = {
    "P1": (0.08, 0.14, "visual_posterior"),
    "N1": (0.14, 0.22, "visual_posterior"),
    "P3": (0.30, 0.60, "frontocentral"),
}


def save_pub(fig, base_path: Path) -> None:
    ensure_parent(str(base_path))
    fig.savefig(f"{base_path}.svg", bbox_inches="tight")
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base_path}.tiff", dpi=600, bbox_inches="tight")


def safe_cache_name(row: pd.Series) -> str:
    key = "|".join(
        [
            str(row.get("subject", "")),
            str(row.get("task", "")),
            str(row.get("raw_task", "")),
            str(row.get("run", "")),
            str(row.get("raw_path", "")),
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    subject = str(row.get("subject", "sub-unknown"))
    task = str(row.get("task", "task"))
    run = str(row.get("run", "run-na") or "run-na")
    return f"{subject}_{task}_{run}_{digest}.npz"


def configured_reject_picks(config: dict, ch_names: list[str]) -> list[int]:
    wanted = []
    for group in ["visual_posterior", "frontocentral"]:
        wanted.extend(config["features"]["channels"].get(group, []))
    wanted_set = set(wanted)
    return [idx for idx, ch in enumerate(ch_names) if ch in wanted_set]


def compute_recording_evoked(row: pd.Series, config: dict, args: argparse.Namespace) -> dict:
    import mne

    raw_path = Path(row["raw_path"])
    events_path = events_path_for_raw(raw_path)
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events file: {events_path}")
    events_tsv = pd.read_csv(events_path, sep="\t")
    if "value" not in events_tsv.columns:
        raise ValueError("events.tsv has no value column")
    visual_events = events_tsv[visual_event_mask(str(row["task"]), events_tsv["value"])].copy()
    if visual_events.empty:
        raise ValueError("No visual events found")

    raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose="ERROR")
    raw.pick("eeg")
    if args.notch_freq:
        raw.notch_filter(freqs=[args.notch_freq], verbose="ERROR")
    raw.filter(l_freq=args.l_freq, h_freq=args.h_freq, verbose="ERROR")
    raw.set_eeg_reference(args.reference, verbose="ERROR")

    events_array = make_events_array(visual_events, float(raw.info["sfreq"]))
    epochs = mne.Epochs(
        raw,
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
    if len(epochs) == 0:
        raise ValueError("No epochs after epoching")

    data = epochs.get_data(copy=False)
    reject_picks = configured_reject_picks(config, epochs.ch_names)
    if args.reject_uv > 0 and reject_picks:
        p2p_uv = np.ptp(data[:, reject_picks, :], axis=2).max(axis=1) * 1e6
        keep = p2p_uv <= args.reject_uv
        data = data[keep]
    else:
        p2p_uv = np.array([])
    if data.shape[0] == 0:
        raise ValueError("No epochs after analysis-channel rejection")

    evoked_uv = data.mean(axis=0).astype(np.float32) * 1e6
    return {
        "evoked_uv": evoked_uv,
        "times": epochs.times.astype(np.float32),
        "ch_names": np.asarray(epochs.ch_names),
        "n_visual_events": int(len(events_array)),
        "n_epochs_kept": int(data.shape[0]),
        "sfreq": float(raw.info["sfreq"]),
        "p2p_uv_median": float(np.median(p2p_uv)) if len(p2p_uv) else np.nan,
    }


def compute_cache(qc: pd.DataFrame, metadata: pd.DataFrame, config: dict, args: argparse.Namespace) -> pd.DataFrame:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = metadata.drop_duplicates("subject").set_index("subject")
    rows = qc[
        (qc["status"].eq("ok"))
        & (qc["task"].isin(["CCD", "SuS"]))
        & (pd.to_numeric(qc["n_epochs_kept"], errors="coerce") > 0)
    ].copy()
    if args.max_recordings > 0:
        rows = rows.head(args.max_recordings).copy()
    if args.n_shards < 1:
        raise ValueError("--n-shards must be >= 1")
    if not 0 <= args.shard_index < args.n_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < n_shards")
    if args.n_shards > 1:
        shard_mask = np.arange(len(rows)) % args.n_shards == args.shard_index
        rows = rows.loc[shard_mask].copy()

    status_rows = []
    total = len(rows)
    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        cache_path = cache_dir / safe_cache_name(row)
        base = {
            "subject": row.get("subject", ""),
            "task": row.get("task", ""),
            "raw_task": row.get("raw_task", ""),
            "run": row.get("run", ""),
            "raw_path": row.get("raw_path", ""),
            "cache_path": str(cache_path),
        }
        if cache_path.exists() and cache_path.stat().st_size > 0 and args.skip_existing:
            status_rows.append({**base, "status": "skipped_existing", "error": ""})
            continue
        try:
            result = compute_recording_evoked(row, config, args)
            subject = str(row["subject"])
            age_group = meta.loc[subject, "age_group"] if subject in meta.index else ""
            age = meta.loc[subject, "age"] if subject in meta.index else np.nan
            sex = meta.loc[subject, "sex"] if subject in meta.index else ""
            np.savez(
                cache_path,
                evoked_uv=result["evoked_uv"],
                times=result["times"],
                ch_names=result["ch_names"],
                subject=subject,
                task=str(row["task"]),
                raw_task=str(row.get("raw_task", "")),
                run=str(row.get("run", "")),
                age_group=str(age_group),
                age=float(age) if pd.notna(age) else np.nan,
                sex=str(sex),
                raw_path=str(row["raw_path"]),
                n_visual_events=result["n_visual_events"],
                n_epochs_kept=result["n_epochs_kept"],
                sfreq=result["sfreq"],
                p2p_uv_median=result["p2p_uv_median"],
            )
            status_rows.append(
                {
                    **base,
                    "status": "ok",
                    "n_visual_events": result["n_visual_events"],
                    "n_epochs_kept": result["n_epochs_kept"],
                    "n_channels": result["evoked_uv"].shape[0],
                    "n_times": result["evoked_uv"].shape[1],
                    "error": "",
                }
            )
        except Exception as exc:
            status_rows.append({**base, "status": "failed", "error": str(exc)})
        if args.progress_every and i % args.progress_every == 0:
            ok = sum(1 for item in status_rows if item["status"] in {"ok", "skipped_existing"})
            print(f"processed {i}/{total}; available caches {ok}; latest={base['subject']} {base['task']} {base['run']}", flush=True)
    return pd.DataFrame(status_rows)


def load_cached_recordings(cache_dir: Path) -> tuple[dict, object, np.ndarray, list[str], pd.DataFrame]:
    import mne

    subject_task_accum: dict[tuple[str, str], dict] = {}
    info = None
    master_ch_names = None
    times = None
    rows = []
    for cache_path in sorted(cache_dir.glob("*.npz")):
        try:
            payload = np.load(cache_path, allow_pickle=False)
        except Exception:
            continue
        subject = str(payload["subject"])
        task = str(payload["task"])
        age_group = str(payload["age_group"])
        if age_group not in AGE_GROUPS or task not in {"CCD", "SuS"}:
            continue
        evoked = payload["evoked_uv"].astype(np.float64)
        ch_names = [str(ch) for ch in payload["ch_names"]]
        if master_ch_names is None:
            master_ch_names = ch_names
            times = payload["times"].astype(float)
            raw_path = str(payload["raw_path"])
            raw = mne.io.read_raw_eeglab(raw_path, preload=False, verbose="ERROR")
            raw.pick("eeg")
            info = raw.info.copy()
        if ch_names != master_ch_names:
            if set(ch_names) != set(master_ch_names):
                continue
            order = [ch_names.index(ch) for ch in master_ch_names]
            evoked = evoked[order]
        weight = float(payload["n_epochs_kept"])
        key = (subject, task)
        if key not in subject_task_accum:
            subject_task_accum[key] = {
                "sum": evoked * weight,
                "weight": weight,
                "age_group": age_group,
                "age": float(payload["age"]),
                "sex": str(payload["sex"]),
            }
        else:
            subject_task_accum[key]["sum"] += evoked * weight
            subject_task_accum[key]["weight"] += weight
        rows.append(
            {
                "subject": subject,
                "task": task,
                "age_group": age_group,
                "cache_path": str(cache_path),
                "n_epochs_kept": int(payload["n_epochs_kept"]),
                "n_visual_events": int(payload["n_visual_events"]),
            }
        )

    subject_task = {}
    for key, value in subject_task_accum.items():
        subject_task[key] = {
            "evoked": value["sum"] / value["weight"],
            "n_epochs": value["weight"],
            "age_group": value["age_group"],
            "age": value["age"],
            "sex": value["sex"],
        }
    return subject_task, info, times, master_ch_names or [], pd.DataFrame(rows)


def build_components(subject_task: dict) -> dict:
    components = {component: [] for component in COMPONENTS}
    subjects = sorted({subject for subject, _ in subject_task.keys()})
    for subject in subjects:
        ccd = subject_task.get((subject, "CCD"))
        sus = subject_task.get((subject, "SuS"))
        if ccd is None or sus is None:
            continue
        components["general"].append(
            {
                "subject": subject,
                "age_group": ccd["age_group"],
                "evoked": (ccd["evoked"] + sus["evoked"]) / 2.0,
            }
        )
        components["specific_sus_minus_ccd"].append(
            {
                "subject": subject,
                "age_group": ccd["age_group"],
                "evoked": (sus["evoked"] - ccd["evoked"]) / np.sqrt(2.0),
            }
        )
    return components


def group_means(components: dict) -> tuple[dict, pd.DataFrame]:
    means = {}
    rows = []
    for component, entries in components.items():
        for age_group in AGE_GROUPS:
            vals = [entry["evoked"] for entry in entries if entry["age_group"] == age_group]
            if not vals:
                continue
            means[(component, age_group)] = np.mean(np.stack(vals), axis=0)
            rows.append({"component": component, "age_group": age_group, "n_subjects": len(vals)})
    return means, pd.DataFrame(rows)


def channel_indices(ch_names: list[str], wanted: list[str]) -> list[int]:
    wanted_set = set(wanted)
    return [idx for idx, ch in enumerate(ch_names) if ch in wanted_set]


def source_tables(means: dict, times: np.ndarray, ch_names: list[str], config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    waveform_rows = []
    groups = {
        "visual_posterior": channel_indices(ch_names, config["features"]["channels"]["visual_posterior"]),
        "frontocentral": channel_indices(ch_names, config["features"]["channels"]["frontocentral"]),
    }
    for (component, age_group), evoked in means.items():
        for group_name, picks in groups.items():
            wave = evoked[picks].mean(axis=0)
            for t, amp in zip(times, wave):
                waveform_rows.append(
                    {
                        "component": component,
                        "age_group": age_group,
                        "channel_group": group_name,
                        "time_s": float(t),
                        "amplitude_uv": float(amp),
                    }
                )
    topomap_rows = []
    contrast_rows = []
    for (component, age_group), evoked in means.items():
        for feature, (tmin, tmax, _) in FEATURE_WINDOWS.items():
            mask = (times >= tmin) & (times <= tmax)
            values = evoked[:, mask].mean(axis=1)
            for ch, amp in zip(ch_names, values):
                topomap_rows.append(
                    {
                        "component": component,
                        "age_group": age_group,
                        "feature": feature,
                        "channel": ch,
                        "amplitude_uv": float(amp),
                    }
                )
    topomap_df = pd.DataFrame(topomap_rows)
    for component in COMPONENTS:
        for feature in FEATURE_WINDOWS:
            child = topomap_df[
                (topomap_df["component"].eq(component))
                & (topomap_df["age_group"].eq("child_5_9"))
                & (topomap_df["feature"].eq(feature))
            ].set_index("channel")["amplitude_uv"]
            youth = topomap_df[
                (topomap_df["component"].eq(component))
                & (topomap_df["age_group"].eq("youth_15_21"))
                & (topomap_df["feature"].eq(feature))
            ].set_index("channel")["amplitude_uv"]
            common = child.index.intersection(youth.index)
            for ch in common:
                contrast_rows.append(
                    {
                        "component": component,
                        "contrast": "youth_15_21_minus_child_5_9",
                        "feature": feature,
                        "channel": ch,
                        "delta_uv": float(youth.loc[ch] - child.loc[ch]),
                    }
                )
    return pd.DataFrame(waveform_rows), topomap_df, pd.DataFrame(contrast_rows)


def all_channel_waveform_source(means: dict, times: np.ndarray, ch_names: list[str]) -> pd.DataFrame:
    rows = []
    for (component, age_group), evoked in means.items():
        for ch_idx, ch_name in enumerate(ch_names):
            for t, amp in zip(times, evoked[ch_idx]):
                rows.append(
                    {
                        "component": component,
                        "age_group": age_group,
                        "channel": ch_name,
                        "time_s": float(t),
                        "amplitude_uv": float(amp),
                    }
                )
    return pd.DataFrame(rows)


def plot_waveforms(waveforms: pd.DataFrame, out_base: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.8), sharex=True)
    groups = ["visual_posterior", "frontocentral"]
    group_labels = {"visual_posterior": "Visual posterior", "frontocentral": "Frontocentral"}
    for r, component in enumerate(COMPONENTS):
        for c, group in enumerate(groups):
            ax = axes[r, c]
            sub = waveforms[(waveforms["component"].eq(component)) & (waveforms["channel_group"].eq(group))]
            for age_group in AGE_GROUPS:
                line = sub[sub["age_group"].eq(age_group)]
                ax.plot(line["time_s"], line["amplitude_uv"], color=AGE_COLORS[age_group], linewidth=1.4, label=AGE_LABELS[age_group])
            ax.axvline(0, color="#777777", linewidth=0.8)
            ax.axhline(0, color="#D8D8D8", linewidth=0.7)
            if group == "visual_posterior":
                ax.axvspan(0.08, 0.14, color="#8FB8DE", alpha=0.13, linewidth=0)
                ax.axvspan(0.14, 0.22, color="#B64342", alpha=0.10, linewidth=0)
                ax.text(0.105, 0.97, "P1", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6)
                ax.text(0.18, 0.97, "N1", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6)
            else:
                ax.axvspan(0.30, 0.60, color="#D9A441", alpha=0.12, linewidth=0)
                ax.text(0.45, 0.97, "P3", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6)
            ax.set_xlim(-0.1, 0.65)
            ax.set_title(f"{COMPONENT_LABELS[component]}: {group_labels[group]}")
            if r == 1:
                ax.set_xlabel("Time from visual event (s)")
            if c == 0:
                ax.set_ylabel("Amplitude (uV)")
            if r == 0 and c == 1:
                ax.legend(loc="upper right", fontsize=6, frameon=False, title="Age")
    fig.suptitle("Full-channel ERP waveforms by age group", y=1.02, fontsize=9, fontweight="bold")
    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    save_pub(fig, out_base)
    plt.close(fig)


def plot_butterfly_waveforms(means: dict, times: np.ndarray, out_base: Path) -> None:
    values = np.concatenate([evoked.ravel() for evoked in means.values()])
    ylim = max(1.0, float(np.nanpercentile(np.abs(values), 99.5)))
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.9), sharex=True, sharey=True)
    for r, component in enumerate(COMPONENTS):
        for c, age_group in enumerate(AGE_GROUPS):
            ax = axes[r, c]
            evoked = means[(component, age_group)]
            ax.plot(times, evoked.T, color=AGE_COLORS[age_group], alpha=0.16, linewidth=0.35)
            ax.axvline(0, color="#777777", linewidth=0.8)
            ax.axhline(0, color="#D8D8D8", linewidth=0.7)
            ax.axvspan(0.08, 0.14, color="#8FB8DE", alpha=0.10, linewidth=0)
            ax.axvspan(0.14, 0.22, color="#B64342", alpha=0.08, linewidth=0)
            ax.axvspan(0.30, 0.60, color="#D9A441", alpha=0.10, linewidth=0)
            ax.set_xlim(-0.1, 0.65)
            ax.set_ylim(-ylim, ylim)
            ax.set_title(f"{COMPONENT_LABELS[component]}\nAge {AGE_LABELS[age_group]}", fontsize=7)
            if r == 1:
                ax.set_xlabel("Time from visual event (s)")
            if c == 0:
                ax.set_ylabel("Amplitude (uV)")
    fig.suptitle("Full-channel ERP butterfly waveforms (129 channels)", y=0.985, fontsize=9, fontweight="bold")
    fig.tight_layout(h_pad=1.0, w_pad=0.9)
    save_pub(fig, out_base)
    plt.close(fig)


def plot_topomaps(contrasts: pd.DataFrame, info, ch_names: list[str], out_base: Path) -> None:
    import mne

    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.9))
    abs_delta = np.abs(contrasts["delta_uv"].to_numpy(dtype=float))
    global_vmax = max(0.1, float(np.nanpercentile(abs_delta, 98)))
    last_im = None
    for r, component in enumerate(COMPONENTS):
        comp_df = contrasts[contrasts["component"].eq(component)]
        for c, feature in enumerate(["P1", "N1", "P3"]):
            ax = axes[r, c]
            row = comp_df[comp_df["feature"].eq(feature)].set_index("channel").reindex(ch_names)
            data = row["delta_uv"].to_numpy(dtype=float)
            try:
                last_im, _ = mne.viz.plot_topomap(
                    data,
                    info,
                    axes=ax,
                    show=False,
                    contours=0,
                    cmap="RdBu_r",
                    vlim=(-global_vmax, global_vmax),
                    sensors=False,
                    names=None,
                )
            except TypeError:
                last_im, _ = mne.viz.plot_topomap(
                    data,
                    info,
                    axes=ax,
                    show=False,
                    contours=0,
                    cmap="RdBu_r",
                    sensors=False,
                    names=None,
                )
                last_im.set_clim(-global_vmax, global_vmax)
            component_title = "General" if component == "general" else "Specific SuS-CCD"
            ax.set_title(f"{component_title}\n{feature}, 15-21 - 5-9", fontsize=7, pad=2)
    fig.subplots_adjust(left=0.04, right=0.88, bottom=0.06, top=0.86, wspace=0.22, hspace=0.34)
    cax = fig.add_axes([0.91, 0.18, 0.025, 0.58])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("Delta amplitude (uV; 98th percentile scale)", rotation=270, labelpad=10)
    fig.suptitle("Full-channel topographies of developmental ERP contrasts", y=0.965, fontsize=9, fontweight="bold")
    save_pub(fig, out_base)
    plt.close(fig)


def aggregate_and_plot(config: dict, args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    subject_task, info, times, ch_names, recording_summary = load_cached_recordings(cache_dir)
    components = build_components(subject_task)
    means, sample_summary = group_means(components)
    waveforms, topomap_values, topomap_contrasts = source_tables(means, times, ch_names, config)
    all_waveforms = all_channel_waveform_source(means, times, ch_names)

    recording_summary.to_csv(out_dir / "full_channel_cached_recording_summary.csv", index=False)
    sample_summary.to_csv(out_dir / "full_channel_erp_sample_summary.csv", index=False)
    waveforms.to_csv(out_dir / "full_channel_erp_group_waveforms.csv", index=False)
    all_waveforms.to_csv(out_dir / "full_channel_erp_all_channel_waveforms.csv", index=False)
    topomap_values.to_csv(out_dir / "full_channel_erp_topomap_window_values.csv", index=False)
    topomap_contrasts.to_csv(out_dir / "full_channel_erp_topomap_age_contrasts.csv", index=False)
    with open(out_dir / "full_channel_erp_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"n_channels": len(ch_names), "ch_names": ch_names, "n_times": len(times)}, f, indent=2)

    plot_waveforms(waveforms, figures_dir / "full_channel_erp_waveforms_general_specific_age_groups")
    plot_butterfly_waveforms(means, times, figures_dir / "full_channel_erp_butterfly_waveforms_general_specific_age_groups")
    plot_topomaps(topomap_contrasts, info, ch_names, figures_dir / "full_channel_erp_topomaps_general_specific_age_contrasts")
    print(sample_summary.to_string(index=False))
    print(f"cached recordings used: {len(recording_summary)}")
    print(f"channels: {len(ch_names)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--qc", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-recordings", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--notch-freq", type=float, default=50.0)
    parser.add_argument("--reference", default="average")
    parser.add_argument("--reject-uv", type=float, default=250.0)
    parser.add_argument("--tmin", type=float, default=-0.2)
    parser.add_argument("--tmax", type=float, default=0.8)
    parser.add_argument("--baseline-tmin", type=float, default=-0.2)
    parser.add_argument("--baseline-tmax", type=float, default=0.0)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.compute:
        qc = pd.read_csv(args.qc)
        metadata = pd.read_csv(args.metadata)
        status = compute_cache(qc, metadata, config, args)
        status_name = "full_channel_evoked_cache_status.csv"
        if args.n_shards > 1:
            status_name = f"full_channel_evoked_cache_status_shard{args.shard_index:02d}_of_{args.n_shards:02d}.csv"
        status_path = Path(args.out_dir) / status_name
        ensure_parent(str(status_path))
        status.to_csv(status_path, index=False)
        print(status["status"].value_counts(dropna=False).to_string())
    if args.aggregate:
        aggregate_and_plot(config, args)


if __name__ == "__main__":
    main()

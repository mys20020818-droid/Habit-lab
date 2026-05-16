from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from s00_utils_common import ensure_parent, load_config


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
    if task == "SL":
        return values.str.contains(r"dot_no\d+_ON", regex=True)
    if task == "SyS":
        return values.eq("newPage")
    return values.str.contains(r"stim_ON|contrastTrial_start|newPage|dot_no\d+_ON", regex=True)


def output_path(derivatives_root: Path, row: pd.Series) -> Path:
    run = str(row.get("run", "") or "run-na")
    raw_task = str(row.get("raw_task", row.get("task", "")))
    subject = str(row["subject"])
    return (
        derivatives_root
        / "visual_epochs"
        / subject
        / f"{subject}_task-{raw_task}_{run}_visual-epo.fif"
    )


def configured_channel_picks(config: dict, groups: list[str], raw_ch_names: list[str]) -> list[str]:
    if not groups:
        return raw_ch_names
    channel_config = config.get("features", {}).get("channels", {})
    picks: list[str] = []
    for group in groups:
        for channel in channel_config.get(group, []):
            if channel in raw_ch_names and channel not in picks:
                picks.append(channel)
    return picks


def make_events_array(events: pd.DataFrame, raw_sfreq: float) -> np.ndarray:
    if "sample" in events.columns:
        samples = pd.to_numeric(events["sample"], errors="coerce")
    else:
        onset = pd.to_numeric(events["onset"], errors="coerce")
        samples = onset * raw_sfreq
    samples = samples.dropna().astype(int)
    return np.column_stack(
        [
            samples.to_numpy(),
            np.zeros(len(samples), dtype=int),
            np.ones(len(samples), dtype=int),
        ]
    )


def preprocess_one(row: pd.Series, config: dict, derivatives_root: Path, args: argparse.Namespace) -> dict:
    import mne

    raw_path = Path(row["raw_path"])
    events_path = events_path_for_raw(raw_path)
    out_path = output_path(derivatives_root, row)
    base = {
        "subject": row.get("subject", ""),
        "task": row.get("task", ""),
        "raw_task": row.get("raw_task", ""),
        "run": row.get("run", ""),
        "release": row.get("release", ""),
        "raw_path": str(raw_path),
        "events_path": str(events_path),
        "epochs_path": str(out_path),
    }

    try:
        if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            return {
                **base,
                "status": "skipped_existing",
                "sfreq": None,
                "n_channels": None,
                "n_visual_events": None,
                "n_epochs_kept": None,
                "drop_fraction": None,
                "p2p_uv_median": None,
                "p2p_uv_p90": None,
                "p2p_uv_max": None,
                "error": "",
            }
        if not events_path.exists():
            raise FileNotFoundError(f"Missing events file: {events_path}")
        events_tsv = pd.read_csv(events_path, sep="\t")
        if "value" not in events_tsv.columns:
            raise ValueError("events.tsv has no value column")

        mask = visual_event_mask(str(row.get("task", "")), events_tsv["value"])
        visual_events = events_tsv[mask].copy()
        if len(visual_events) == 0:
            raise ValueError("No visual-onset events found")

        raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose="ERROR")
        raw.pick("eeg")
        picks = configured_channel_picks(config, args.channel_groups, raw.ch_names)
        if args.channel_groups:
            if not picks:
                raise ValueError(f"No configured channels found for groups: {args.channel_groups}")
            raw.pick(picks)

        if args.max_channels and len(raw.ch_names) > args.max_channels:
            raw.pick(raw.ch_names[: args.max_channels])

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
        data = epochs.get_data(copy=False)
        p2p_uv = np.ptp(data, axis=2).max(axis=1) * 1e6 if len(epochs) else np.array([])
        reject = None
        if args.reject_uv > 0:
            reject = {"eeg": args.reject_uv * 1e-6}
            epochs.drop_bad(reject=reject, verbose="ERROR")
        ensure_parent(out_path)
        epochs.save(out_path, overwrite=True, verbose="ERROR")

        n_input = len(events_array)
        n_kept = len(epochs)
        return {
            **base,
            "status": "ok",
            "sfreq": float(raw.info["sfreq"]),
            "n_channels": len(raw.ch_names),
            "n_visual_events": n_input,
            "n_epochs_kept": n_kept,
            "drop_fraction": 1.0 - (n_kept / n_input if n_input else 0.0),
            "p2p_uv_median": float(np.median(p2p_uv)) if len(p2p_uv) else None,
            "p2p_uv_p90": float(np.quantile(p2p_uv, 0.90)) if len(p2p_uv) else None,
            "p2p_uv_max": float(np.max(p2p_uv)) if len(p2p_uv) else None,
            "error": "",
        }
    except Exception as exc:
        return {
            **base,
            "status": "failed",
            "sfreq": None,
            "n_channels": None,
            "n_visual_events": None,
            "n_epochs_kept": None,
            "drop_fraction": None,
            "p2p_uv_median": None,
            "p2p_uv_p90": None,
            "p2p_uv_max": None,
            "error": str(exc),
        }


def select_rows(manifest: pd.DataFrame, max_recordings: int) -> pd.DataFrame:
    if max_recordings <= 0 or len(manifest) <= max_recordings:
        return manifest
    per_task = max(1, max_recordings // max(1, manifest["task"].nunique()))
    sampled = manifest.groupby("task", group_keys=False).head(per_task)
    if len(sampled) < max_recordings:
        remainder = manifest[~manifest.index.isin(sampled.index)].head(max_recordings - len(sampled))
        sampled = pd.concat([sampled, remainder], ignore_index=False)
    return sampled.head(max_recordings).copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--qc-out", required=True)
    parser.add_argument("--derivatives-root", default=None)
    parser.add_argument("--max-recordings", type=int, default=12)
    parser.add_argument("--tasks", nargs="*", default=["SuS", "CCD", "SL"])
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--notch-freq", type=float, default=60.0)
    parser.add_argument("--reference", default="average")
    parser.add_argument("--reject-uv", type=float, default=250.0)
    parser.add_argument("--tmin", type=float, default=-0.2)
    parser.add_argument("--tmax", type=float, default=0.8)
    parser.add_argument("--baseline-tmin", type=float, default=-0.2)
    parser.add_argument("--baseline-tmax", type=float, default=0.0)
    parser.add_argument("--max-channels", type=int, default=0)
    parser.add_argument("--channel-groups", nargs="*", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    derivatives_root = Path(args.derivatives_root or config["paths"]["derivatives_root"])
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["task"].isin(args.tasks)].copy()
    manifest = select_rows(manifest, args.max_recordings)
    rows = [preprocess_one(row, config, derivatives_root, args) for _, row in manifest.iterrows()]
    qc = pd.DataFrame(rows)
    ensure_parent(args.qc_out)
    qc.to_csv(args.qc_out, index=False)


if __name__ == "__main__":
    main()

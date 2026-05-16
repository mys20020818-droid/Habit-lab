from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from s00_utils_common import ensure_parent, load_config


def channel_group(config: dict, name: str) -> list[str]:
    return list(config.get("features", {}).get("channels", {}).get(name, []))


def extract_one(row: pd.Series, config: dict) -> list[dict]:
    import mne

    if row.get("status") != "ok":
        return []
    epochs_path = Path(row["epochs_path"])
    if not epochs_path.exists():
        return []

    epochs = mne.read_epochs(epochs_path, preload=True, verbose="ERROR")
    rows: list[dict] = []
    if len(epochs) == 0:
        return rows

    times = epochs.times
    data = epochs.get_data(copy=False) * 1e6
    ch_names = epochs.ch_names

    for feature, spec in config["features"]["windows"].items():
        group_name = spec["channel_group"]
        wanted = channel_group(config, group_name)
        picks = [idx for idx, ch in enumerate(ch_names) if ch in wanted]
        time_mask = (times >= float(spec["tmin"])) & (times <= float(spec["tmax"]))
        if not picks or not time_mask.any():
            continue
        epoch_values = data[:, picks, :][:, :, time_mask].mean(axis=(1, 2))
        odd = epoch_values[::2]
        even = epoch_values[1::2]
        rows.append(
            {
                "subject": row.get("subject", ""),
                "task": row.get("task", ""),
                "raw_task": row.get("raw_task", ""),
                "run": row.get("run", ""),
                "release": row.get("release", ""),
                "feature": feature,
                "channel_group": group_name,
                "n_epochs": len(epoch_values),
                "amplitude_uv": float(np.mean(epoch_values)),
                "amplitude_uv_sd": float(np.std(epoch_values, ddof=1)) if len(epoch_values) > 1 else 0.0,
                "split_odd_uv": float(np.mean(odd)) if len(odd) else np.nan,
                "split_even_uv": float(np.mean(even)) if len(even) else np.nan,
                "epochs_path": str(epochs_path),
            }
        )
    return rows


def aggregate_subject_task(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    weighted = features.copy()
    weighted["weighted_amp"] = weighted["amplitude_uv"] * weighted["n_epochs"]
    weighted["weighted_odd"] = weighted["split_odd_uv"] * weighted["n_epochs"]
    weighted["weighted_even"] = weighted["split_even_uv"] * weighted["n_epochs"]
    grouped = (
        weighted.groupby(["subject", "task", "feature", "channel_group"], dropna=False)
        .agg(
            n_recordings=("epochs_path", "nunique"),
            n_epochs=("n_epochs", "sum"),
            weighted_amp=("weighted_amp", "sum"),
            weighted_odd=("weighted_odd", "sum"),
            weighted_even=("weighted_even", "sum"),
        )
        .reset_index()
    )
    grouped["amplitude_uv"] = grouped["weighted_amp"] / grouped["n_epochs"]
    grouped["split_odd_uv"] = grouped["weighted_odd"] / grouped["n_epochs"]
    grouped["split_even_uv"] = grouped["weighted_even"] / grouped["n_epochs"]
    return grouped.drop(columns=["weighted_amp", "weighted_odd", "weighted_even"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--qc", required=True)
    parser.add_argument("--recording-out", required=True)
    parser.add_argument("--subject-task-out", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    qc = pd.read_csv(args.qc)
    rows: list[dict] = []
    for _, row in qc.iterrows():
        rows.extend(extract_one(row, config))
    features = pd.DataFrame(rows)
    subject_task = aggregate_subject_task(features)

    ensure_parent(args.recording_out)
    ensure_parent(args.subject_task_out)
    features.to_csv(args.recording_out, index=False)
    subject_task.to_csv(args.subject_task_out, index=False)


if __name__ == "__main__":
    main()


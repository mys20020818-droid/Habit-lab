from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from s00_utils_common import ensure_parent


EVENT_STAT_COLUMNS = [
    "subject",
    "task",
    "raw_task",
    "run",
    "release",
    "raw_path",
    "events_path",
    "n_events_total",
    "n_visual_onset",
    "n_contrast_change",
    "n_motor_response",
    "n_feedback_proxy",
    "n_search_array",
    "n_sequence_onset",
    "n_rest_eye_state",
    "n_scene_boundary",
    "status",
]


def raw_stem(path: Path) -> str:
    for suffix in ["_eeg.set", "_eeg.fif", "_eeg.edf", "_eeg.bdf", "_eeg.mff"]:
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def events_path_for_raw(raw_path: Path) -> Path:
    return raw_path.with_name(f"{raw_stem(raw_path)}_events.tsv")


def value_series(events: pd.DataFrame) -> pd.Series:
    if "value" not in events.columns:
        return pd.Series([], dtype=str)
    return events["value"].fillna("").astype(str)


def count_operations(task: str, values: pd.Series, events: pd.DataFrame) -> dict[str, int]:
    visual = pd.Series(False, index=values.index)
    contrast = pd.Series(False, index=values.index)
    motor = pd.Series(False, index=values.index)
    feedback = pd.Series(False, index=values.index)
    search = pd.Series(False, index=values.index)
    sequence = pd.Series(False, index=values.index)
    eye_state = pd.Series(False, index=values.index)
    scene = pd.Series(False, index=values.index)

    if task == "SuS":
        visual = values.eq("stim_ON")
    elif task == "CCD":
        visual = values.eq("contrastTrial_start")
        contrast = values.str.contains(r"(?:left|right)_target", regex=True)
        motor = values.str.contains(r"(?:left|right)_buttonPress", regex=True)
        if "feedback" in events.columns:
            feedback = events["feedback"].fillna("").astype(str).ne("n/a") & motor
    elif task == "SL":
        sequence = values.str.contains(r"dot_no\d+_ON", regex=True)
        visual = sequence
        motor = values.str.contains(r"trialResponse|buttonPress|response", case=False, regex=True)
    elif task == "SyS":
        search = values.eq("newPage")
        visual = search
        motor = values.eq("trialResponse")
    elif task == "RS":
        eye_state = values.str.contains(r"instructed_to(?:Open|Close)Eyes", regex=True)
    elif task == "MW":
        scene = values.eq("boundary")
        visual = values.eq("video_start") | scene
    else:
        visual = values.str.contains(r"stim_ON|contrastTrial_start|newPage|dot_no\d+_ON|video_start", regex=True)
        motor = values.str.contains(r"buttonPress|trialResponse|response", case=False, regex=True)

    return {
        "n_visual_onset": int(visual.sum()),
        "n_contrast_change": int(contrast.sum()),
        "n_motor_response": int(motor.sum()),
        "n_feedback_proxy": int(feedback.sum()),
        "n_search_array": int(search.sum()),
        "n_sequence_onset": int(sequence.sum()),
        "n_rest_eye_state": int(eye_state.sum()),
        "n_scene_boundary": int(scene.sum()),
    }


def summarize_recording(row: pd.Series) -> dict:
    raw_path = Path(row["raw_path"])
    events_path = events_path_for_raw(raw_path)
    base = {col: row.get(col, "") for col in ["subject", "task", "raw_task", "run", "release", "raw_path"]}
    base["events_path"] = str(events_path)
    if not events_path.exists():
        return {
            **base,
            "n_events_total": 0,
            **{col: 0 for col in EVENT_STAT_COLUMNS if col.startswith("n_") and col != "n_events_total"},
            "status": "missing_events",
        }
    try:
        events = pd.read_csv(events_path, sep="\t")
    except Exception as exc:
        return {
            **base,
            "n_events_total": 0,
            **{col: 0 for col in EVENT_STAT_COLUMNS if col.startswith("n_") and col != "n_events_total"},
            "status": f"read_failed:{exc}",
        }
    values = value_series(events)
    counts = count_operations(str(row.get("task", "")), values, events)
    return {
        **base,
        "n_events_total": len(events),
        **counts,
        "status": "ok",
    }


def write_subject_task_summary(recordings: pd.DataFrame, out: str) -> None:
    if recordings.empty:
        summary = pd.DataFrame()
    else:
        numeric_cols = [col for col in recordings.columns if col.startswith("n_")]
        summary = (
            recordings.groupby(["subject", "task", "release"], dropna=False)[numeric_cols]
            .sum()
            .reset_index()
        )
        status = (
            recordings.groupby(["subject", "task", "release"], dropna=False)["status"]
            .apply(lambda s: ";".join(sorted(set(s))))
            .reset_index()
        )
        summary = summary.merge(status, on=["subject", "task", "release"], how="left")
    ensure_parent(out)
    summary.to_csv(out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--recording-out", required=True)
    parser.add_argument("--subject-task-out", required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    rows = [summarize_recording(row) for _, row in manifest.iterrows()]
    recordings = pd.DataFrame(rows).reindex(columns=EVENT_STAT_COLUMNS)

    ensure_parent(args.recording_out)
    recordings.to_csv(args.recording_out, index=False)
    write_subject_task_summary(recordings, args.subject_task_out)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from s00_utils_common import ensure_parent, load_config
from s01_data_build_manifest import release_name_for_root, task_alias_map, canonical_task
from s03_data_classify_age_groups import participants_file_for_root


EEG_SUFFIXES = (".set", ".fif", ".edf", ".bdf", ".mff")
REQUIRED_PARTICIPANT_COLUMNS = [
    "participant_id",
    "sex",
    "age",
    "p_factor",
    "attention",
    "internalizing",
    "externalizing",
]
REQUIRED_EVENTS_COLUMNS = ["onset", "duration"]
REQUIRED_CHANNEL_COLUMNS = ["name", "type", "units"]


def parse_task(path: Path) -> str:
    match = re.search(r"_task-([^_]+)", path.name)
    return match.group(1) if match else ""


def raw_stem(path: Path) -> str:
    for suffix in ["_eeg.set", "_eeg.fif", "_eeg.edf", "_eeg.bdf", "_eeg.mff"]:
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def eeg_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.glob("**/*_eeg.*"):
        if path.is_file() and path.suffix.lower() in EEG_SUFFIXES:
            files.append(path)
    return sorted(files)


def safe_header(path: Path, sep: str = "\t") -> list[str]:
    if not path.exists():
        return []
    try:
        return list(pd.read_csv(path, sep=sep, nrows=0).columns)
    except Exception:
        return []


def metadata_root(root: Path) -> Path | None:
    for candidate in [root, *root.parents]:
        if (candidate / "dataset_description.json").exists():
            return candidate
    return None


def inspect_root(
    root: Path, config: dict, max_header_checks_per_root: int
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    aliases = task_alias_map(config)
    release = release_name_for_root(root)
    meta_root = metadata_root(root)
    participants_path = participants_file_for_root(root)
    files = eeg_files(root)

    task_counts: Counter[str] = Counter()
    raw_task_counts: Counter[str] = Counter()
    ext_counts: Counter[str] = Counter(path.suffix.lower() for path in files)
    missing_rows: list[dict] = []
    zero_byte_eeg = 0
    zero_byte_sidecars = 0
    event_column_problems = 0
    channel_column_problems = 0

    checked_headers = 0
    for eeg_path in files:
        if eeg_path.stat().st_size == 0:
            zero_byte_eeg += 1
        stem = raw_stem(eeg_path)
        raw_task = parse_task(eeg_path)
        task = canonical_task(raw_task, aliases) if raw_task else "missing_task"
        task_counts[task] += 1
        raw_task_counts[raw_task or "missing_task"] += 1

        sidecars = {
            "events": eeg_path.with_name(f"{stem}_events.tsv"),
            "channels": eeg_path.with_name(f"{stem}_channels.tsv"),
            "eeg_json": eeg_path.with_name(f"{stem}_eeg.json"),
        }
        for sidecar_type, sidecar_path in sidecars.items():
            if not sidecar_path.exists():
                missing_rows.append(
                    {
                        "bids_root": str(root),
                        "release": release,
                        "eeg_path": str(eeg_path),
                        "missing_sidecar": sidecar_type,
                        "expected_path": str(sidecar_path),
                    }
                )
            elif sidecar_path.stat().st_size == 0:
                zero_byte_sidecars += 1

        if checked_headers < max_header_checks_per_root:
            event_cols = safe_header(sidecars["events"])
            if event_cols and any(col not in event_cols for col in REQUIRED_EVENTS_COLUMNS):
                event_column_problems += 1
            channel_cols = safe_header(sidecars["channels"])
            if channel_cols and any(col not in channel_cols for col in REQUIRED_CHANNEL_COLUMNS):
                channel_column_problems += 1
            checked_headers += 1

    participants_rows = 0
    missing_participant_columns: list[str] = REQUIRED_PARTICIPANT_COLUMNS.copy()
    age_missing = None
    age_min = None
    age_max = None
    if participants_path and participants_path.exists():
        participants = pd.read_csv(participants_path, sep="\t")
        participants_rows = len(participants)
        missing_participant_columns = [
            col for col in REQUIRED_PARTICIPANT_COLUMNS if col not in participants.columns
        ]
        if "age" in participants.columns:
            age = pd.to_numeric(participants["age"], errors="coerce")
            age_missing = int(age.isna().sum())
            age_min = float(age.min()) if age.notna().any() else None
            age_max = float(age.max()) if age.notna().any() else None

    subject_dirs = [path for path in root.glob("sub-*") if path.is_dir()]
    summary = {
        "bids_root": str(root),
        "release": release,
        "metadata_root": "" if meta_root is None else str(meta_root),
        "dataset_description_at_scan_root": (root / "dataset_description.json").exists(),
        "dataset_description_resolved": meta_root is not None,
        "participants_path": "" if participants_path is None else str(participants_path),
        "participants_rows": participants_rows,
        "missing_participant_columns": ";".join(missing_participant_columns),
        "age_missing": age_missing,
        "age_min": age_min,
        "age_max": age_max,
        "subject_dirs_at_scan_root": len(subject_dirs),
        "eeg_files": len(files),
        "extensions": json.dumps(dict(ext_counts), sort_keys=True),
        "missing_sidecars": len(missing_rows),
        "zero_byte_eeg_files": zero_byte_eeg,
        "zero_byte_sidecars": zero_byte_sidecars,
        "event_header_problems": event_column_problems,
        "channel_header_problems": channel_column_problems,
        "header_files_checked": checked_headers,
        "unmapped_raw_tasks": sum(
            count
            for raw_task, count in raw_task_counts.items()
            if raw_task != "missing_task" and canonical_task(raw_task, aliases) == raw_task
        ),
    }

    task_table = pd.DataFrame(
        [
            {
                "bids_root": str(root),
                "release": release,
                "task": task,
                "n_eeg_files": count,
            }
            for task, count in sorted(task_counts.items())
        ]
    )
    missing = pd.DataFrame(missing_rows)
    return summary, task_table, missing


def write_markdown_report(summary: pd.DataFrame, task_counts: pd.DataFrame, out: str) -> None:
    total_eeg = int(summary["eeg_files"].sum()) if not summary.empty else 0
    total_missing = int(summary["missing_sidecars"].sum()) if not summary.empty else 0
    root_count = len(summary)
    lines = [
        "# HBN-EEG Data Format Check",
        "",
        f"- BIDS roots checked: {root_count}",
        f"- EEG files found: {total_eeg}",
        f"- Missing sidecars: {total_missing}",
        "",
        "## Root Summary",
        "",
        summary[
            [
                "release",
                "subject_dirs_at_scan_root",
                "participants_rows",
                "eeg_files",
                "missing_sidecars",
                "zero_byte_eeg_files",
                "zero_byte_sidecars",
                "unmapped_raw_tasks",
            ]
        ].to_markdown(index=False),
        "",
        "## Task Counts",
        "",
        task_counts.groupby("task", as_index=False)["n_eeg_files"]
        .sum()
        .sort_values("task")
        .to_markdown(index=False),
        "",
    ]
    ensure_parent(out)
    Path(out).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--task-counts-out", required=True)
    parser.add_argument("--missing-sidecars-out", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--max-header-checks-per-root", type=int, default=25)
    args = parser.parse_args()

    config = load_config(args.config)
    summaries: list[dict] = []
    task_tables: list[pd.DataFrame] = []
    missing_tables: list[pd.DataFrame] = []

    for root_text in config["paths"].get("bids_roots", []):
        root = Path(root_text)
        if not root.exists():
            summaries.append(
                {
                    "bids_root": str(root),
                    "release": "missing_root",
                    "metadata_root": "",
                    "dataset_description_at_scan_root": False,
                    "dataset_description_resolved": False,
                    "participants_path": "",
                    "participants_rows": 0,
                    "missing_participant_columns": ";".join(REQUIRED_PARTICIPANT_COLUMNS),
                    "age_missing": None,
                    "age_min": None,
                    "age_max": None,
                    "subject_dirs_at_scan_root": 0,
                    "eeg_files": 0,
                    "extensions": "{}",
                    "missing_sidecars": 0,
                    "zero_byte_eeg_files": 0,
                    "zero_byte_sidecars": 0,
                    "event_header_problems": 0,
                    "channel_header_problems": 0,
                    "header_files_checked": 0,
                    "unmapped_raw_tasks": 0,
                }
            )
            continue
        summary, task_table, missing = inspect_root(root, config, args.max_header_checks_per_root)
        summaries.append(summary)
        task_tables.append(task_table)
        missing_tables.append(missing)

    summary_df = pd.DataFrame(summaries)
    task_counts = pd.concat(task_tables, ignore_index=True) if task_tables else pd.DataFrame()
    missing_sidecars = (
        pd.concat(missing_tables, ignore_index=True)
        if missing_tables
        else pd.DataFrame(columns=["bids_root", "release", "eeg_path", "missing_sidecar", "expected_path"])
    )

    for path, table in [
        (args.summary_out, summary_df),
        (args.task_counts_out, task_counts),
        (args.missing_sidecars_out, missing_sidecars),
    ]:
        ensure_parent(path)
        table.to_csv(path, index=False)
    write_markdown_report(summary_df, task_counts, args.report_out)


if __name__ == "__main__":
    main()

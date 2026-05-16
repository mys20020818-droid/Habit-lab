from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from s00_utils_common import ensure_parent, load_config


EVENT_COLUMNS = [
    "onset",
    "duration",
    "trial_type",
    "HED",
    "operation",
    "subject",
    "task",
    "raw_path",
    "events_path",
    "event_id",
]


def classify_operation(trial_type: str, hed: str, operations: dict) -> str | None:
    haystack = f"{trial_type} {hed}".lower()
    for operation, spec in operations.items():
        for term in spec.get("hed_terms", []):
            if term.lower() in haystack:
                return operation
        for pattern in spec.get("fallback_trial_type_regex", []):
            if re.search(pattern, haystack, flags=re.IGNORECASE):
                return operation
    return None


def events_file_for_raw(raw_path: str) -> Path:
    path = Path(raw_path)
    return path.with_name(path.name.replace("_eeg" + path.suffix, "_events.tsv"))


def load_events_for_row(row: pd.Series, operations: dict) -> pd.DataFrame:
    events_path = events_file_for_raw(row["raw_path"])
    if not events_path.exists():
        return pd.DataFrame()

    events = pd.read_csv(events_path, sep="\t")
    if "trial_type" not in events.columns:
        events["trial_type"] = ""
    if "HED" not in events.columns:
        events["HED"] = ""

    events["operation"] = [
        classify_operation(tt, hed, operations)
        for tt, hed in zip(events["trial_type"].fillna(""), events["HED"].fillna(""))
    ]
    events = events[events["operation"].notna()].copy()
    events["subject"] = row["subject"]
    events["task"] = row["task"]
    events["raw_path"] = row["raw_path"]
    events["events_path"] = str(events_path)
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = pd.read_csv(args.manifest)
    operations = config["events"]["operations"]
    tables = [load_events_for_row(row, operations) for _, row in manifest.iterrows()]
    events = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    if not events.empty:
        primary_operation = config["pilot"]["primary_operation"]
        events = events[events["operation"].eq(primary_operation)].copy()
        events["event_id"] = (
            events["subject"].astype(str)
            + "_"
            + events["task"].astype(str)
            + "_"
            + events.groupby(["subject", "task"]).cumcount().astype(str)
        )
    else:
        events = pd.DataFrame(columns=EVENT_COLUMNS)

    ensure_parent(args.out)
    events.reindex(columns=EVENT_COLUMNS).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

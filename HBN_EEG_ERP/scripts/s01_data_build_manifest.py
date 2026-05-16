from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from s00_utils_common import ensure_parent, load_config, normalize_subject, normalize_task


def find_raw_files(bids_root: Path) -> list[Path]:
    patterns = ("*_eeg.set", "*_eeg.fif", "*_eeg.edf", "*_eeg.bdf", "*_eeg.mff")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(
            path for path in bids_root.glob(f"**/{pattern}") if any(parent.name.startswith("sub-") for parent in path.parents)
        )
    return sorted(set(files))


def task_alias_map(config: dict) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for canonical, spec in config.get("tasks", {}).items():
        aliases[canonical.lower()] = canonical
        for alias in spec.get("aliases", []):
            aliases[str(alias).lower()] = canonical
    return aliases


def canonical_task(raw_task: str, aliases: dict[str, str]) -> str:
    return aliases.get(raw_task.lower(), raw_task)


def release_name_for_root(bids_root: Path) -> str:
    for candidate_root in [bids_root, *bids_root.parents]:
        desc = candidate_root / "dataset_description.json"
        if desc.exists():
            try:
                payload = json.loads(desc.read_text(encoding="utf-8-sig"))
                return str(payload.get("Name", candidate_root.name))
            except json.JSONDecodeError:
                return candidate_root.name
    return bids_root.name


def parse_bids_like_path(path: Path, aliases: dict[str, str]) -> dict[str, str]:
    parts = path.name.split("_")
    record = {
        "subject": "",
        "session": "",
        "task": "",
        "raw_task": "",
        "run": "",
        "raw_path": str(path),
    }
    for part in parts:
        if part.startswith("sub-"):
            record["subject"] = normalize_subject(part.replace("sub-", ""))
        elif part.startswith("ses-"):
            record["session"] = part
        elif part.startswith("task-"):
            raw_task = normalize_task(part)
            record["raw_task"] = raw_task
            record["task"] = canonical_task(raw_task, aliases)
        elif part.startswith("run-"):
            record["run"] = part
    if not record["subject"]:
        for parent in path.parents:
            if parent.name.startswith("sub-"):
                record["subject"] = parent.name
                break
    return record


def load_phenotype(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    phenotype_path = Path(path)
    if not phenotype_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(phenotype_path)
    if "subject" not in df.columns:
        for candidate in ("participant_id", "src_subject_id", "EID"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "subject"})
                break
    if "subject" in df.columns:
        df["subject"] = df["subject"].map(normalize_subject)
    return df


def apply_pilot_filter(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    pilot = config.get("pilot", {})
    if not pilot.get("enabled", False) or df.empty:
        return df

    tasks = set(pilot.get("include_tasks", []))
    if tasks and "task" in df.columns:
        df = df[df["task"].isin(tasks)].copy()

    max_subjects = pilot.get("max_subjects")
    if max_subjects:
        subjects = sorted(df["subject"].dropna().unique())[: int(max_subjects)]
        df = df[df["subject"].isin(subjects)].copy()
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-pilot", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    rows: list[dict[str, str]] = []
    aliases = task_alias_map(config)

    for root in config["paths"].get("bids_roots", []):
        bids_root = Path(root)
        if not bids_root.exists():
            continue
        release_name = release_name_for_root(bids_root)
        for raw_file in find_raw_files(bids_root):
            record = parse_bids_like_path(raw_file, aliases)
            record["bids_root"] = str(bids_root)
            record["release"] = release_name
            rows.append(record)

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        manifest = pd.DataFrame(
            columns=["subject", "session", "task", "raw_task", "run", "raw_path", "bids_root", "release"]
        )
    else:
        manifest = manifest.drop_duplicates(
            subset=["subject", "session", "task", "raw_task", "run"],
            keep="first",
        ).copy()

    phenotype = load_phenotype(config["paths"].get("phenotype_file"))
    if not phenotype.empty and "subject" in phenotype.columns:
        manifest = manifest.merge(phenotype, on="subject", how="left")

    if args.no_pilot:
        config = {**config, "pilot": {**config.get("pilot", {}), "enabled": False}}
    manifest = apply_pilot_filter(manifest, config)
    ensure_parent(args.out)
    manifest.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

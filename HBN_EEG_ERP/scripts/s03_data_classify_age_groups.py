from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from s00_utils_common import ensure_parent, load_config, normalize_subject


AGE_GROUPS = [
    ("child_5_9", 5.0, 10.0),
    ("adolescent_10_14", 10.0, 15.0),
    ("youth_15_21", 15.0, 22.0),
]


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


def participants_file_for_root(bids_root: Path) -> Path | None:
    for candidate_root in [bids_root, *bids_root.parents]:
        path = candidate_root / "participants.tsv"
        if path.exists():
            return path
    return None


def age_group(age: object) -> str:
    if pd.isna(age):
        return "missing_age"
    try:
        value = float(age)
    except (TypeError, ValueError):
        return "missing_age"
    for label, lower, upper in AGE_GROUPS:
        if lower <= value < upper:
            return label
    return "outside_5_21"


def load_participants(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, str]] = []

    for root in config["paths"].get("bids_roots", []):
        bids_root = Path(root)
        release = release_name_for_root(bids_root)
        participants_path = participants_file_for_root(bids_root)
        inventory_rows.append(
            {
                "bids_root": str(bids_root),
                "release": release,
                "participants_path": "" if participants_path is None else str(participants_path),
                "has_participants": str(participants_path is not None),
            }
        )
        if participants_path is None:
            continue

        df = pd.read_csv(participants_path, sep="\t")
        if "participant_id" in df.columns and "subject" not in df.columns:
            df = df.rename(columns={"participant_id": "subject"})
        if "subject" not in df.columns:
            continue

        keep = [
            col
            for col in [
                "subject",
                "release_number",
                "sex",
                "age",
                "full_pheno",
                "p_factor",
                "attention",
                "internalizing",
                "externalizing",
            ]
            if col in df.columns
        ]
        df = df[keep].copy()
        df["subject"] = df["subject"].map(normalize_subject)
        df["bids_root"] = str(bids_root)
        df["release"] = release
        rows.append(df)

    participants = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not participants.empty:
        participants["age"] = pd.to_numeric(participants["age"], errors="coerce")
        participants["age_group"] = participants["age"].map(age_group)
        participants = participants.sort_values(["release", "subject"]).drop_duplicates(
            subset=["subject"], keep="first"
        )

    inventory = pd.DataFrame(inventory_rows)
    return participants, inventory


def load_downloaded_subjects(config: dict, participants: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for root in config["paths"].get("bids_roots", []):
        bids_root = Path(root)
        release = release_name_for_root(bids_root)
        if not bids_root.exists():
            continue
        for subject_dir in sorted(bids_root.glob("sub-*")):
            if subject_dir.is_dir():
                rows.append(
                    {
                        "subject": normalize_subject(subject_dir.name.replace("sub-", "")),
                        "bids_root": str(bids_root),
                        "release": release,
                    }
                )

    downloaded = pd.DataFrame(rows)
    if downloaded.empty:
        return pd.DataFrame(
            columns=["subject", "bids_root", "release", "age", "age_group", "sex", "full_pheno"]
        )

    participant_cols = [
        col
        for col in [
            "subject",
            "age",
            "age_group",
            "sex",
            "full_pheno",
            "p_factor",
            "attention",
            "internalizing",
            "externalizing",
        ]
        if col in participants.columns
    ]
    if participant_cols:
        downloaded = downloaded.merge(
            participants[participant_cols].drop_duplicates("subject"),
            on="subject",
            how="left",
        )
    if "age_group" not in downloaded.columns:
        downloaded["age_group"] = "missing_age"
    downloaded["age_group"] = downloaded["age_group"].fillna("missing_age")
    return downloaded.sort_values(["age_group", "subject"])


def build_subject_summary(participants: pd.DataFrame) -> pd.DataFrame:
    if participants.empty:
        return pd.DataFrame(
            columns=["age_group", "n_subjects", "age_min", "age_mean", "age_max", "n_full_pheno"]
        )
    summary = (
        participants.groupby("age_group", dropna=False)
        .agg(
            n_subjects=("subject", "nunique"),
            age_min=("age", "min"),
            age_mean=("age", "mean"),
            age_max=("age", "max"),
            n_full_pheno=("full_pheno", lambda s: (s.astype(str).str.lower() == "yes").sum())
            if "full_pheno" in participants.columns
            else ("subject", "size"),
        )
        .reset_index()
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=False)
    parser.add_argument("--participants-out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--downloaded-out", required=True)
    parser.add_argument("--downloaded-summary-out", required=True)
    parser.add_argument("--inventory-out", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    participants, inventory = load_participants(config)
    summary = build_subject_summary(participants)
    downloaded = load_downloaded_subjects(config, participants)
    downloaded_summary = build_subject_summary(downloaded)

    manifest_with_age = pd.DataFrame()
    if args.manifest and Path(args.manifest).exists():
        manifest = pd.read_csv(args.manifest)
        if not manifest.empty and not participants.empty:
            participant_cols = [
                col
                for col in [
                    "subject",
                    "age",
                    "age_group",
                    "sex",
                    "full_pheno",
                    "p_factor",
                    "attention",
                    "internalizing",
                    "externalizing",
                ]
                if col in participants.columns
            ]
            manifest_with_age = manifest.merge(
                participants[participant_cols].drop_duplicates("subject"),
                on="subject",
                how="left",
            )
            manifest_with_age["age_group"] = manifest_with_age["age_group"].fillna("missing_age")
        else:
            manifest_with_age = manifest.copy()
            if "age_group" not in manifest_with_age.columns:
                manifest_with_age["age_group"] = "missing_age"

    for path, table in [
        (args.participants_out, participants),
        (args.manifest_out, manifest_with_age),
        (args.summary_out, summary),
        (args.downloaded_out, downloaded),
        (args.downloaded_summary_out, downloaded_summary),
        (args.inventory_out, inventory),
    ]:
        ensure_parent(path)
        table.to_csv(path, index=False)


if __name__ == "__main__":
    main()

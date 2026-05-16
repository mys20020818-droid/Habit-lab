from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from s00_utils_common import ensure_parent


def raw_stem(path: Path) -> str:
    for suffix in ["_eeg.set", "_eeg.fif", "_eeg.edf", "_eeg.bdf", "_eeg.mff"]:
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def sidecar_paths(raw_path: Path) -> dict[str, Path]:
    stem = raw_stem(raw_path)
    return {
        "events": raw_path.with_name(f"{stem}_events.tsv"),
        "channels": raw_path.with_name(f"{stem}_channels.tsv"),
        "eeg_json": raw_path.with_name(f"{stem}_eeg.json"),
    }


def issue_for_recording(raw_path_text: str) -> str:
    raw_path = Path(raw_path_text)
    issues: list[str] = []
    if not raw_path.exists():
        issues.append("missing_eeg_file")
        return ";".join(issues)
    if raw_path.stat().st_size == 0:
        issues.append("zero_byte_eeg")
    for name, path in sidecar_paths(raw_path).items():
        if not path.exists():
            issues.append(f"missing_{name}")
        elif path.stat().st_size == 0:
            issues.append(f"zero_byte_{name}")
    return ";".join(issues)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clean-out", required=True)
    parser.add_argument("--excluded-out", required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if manifest.empty:
        clean = manifest.copy()
        excluded = manifest.copy()
        excluded["exclusion_reason"] = []
    else:
        manifest["exclusion_reason"] = manifest["raw_path"].map(issue_for_recording)
        clean = manifest[manifest["exclusion_reason"].eq("")].copy()
        excluded = manifest[~manifest["exclusion_reason"].eq("")].copy()

    ensure_parent(args.clean_out)
    ensure_parent(args.excluded_out)
    clean.to_csv(args.clean_out, index=False)
    excluded.to_csv(args.excluded_out, index=False)


if __name__ == "__main__":
    main()


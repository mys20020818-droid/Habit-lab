from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from s00_utils_common import ensure_parent


def read_one(row: pd.Series) -> dict:
    raw_path = Path(row["raw_path"])
    base = {
        "subject": row.get("subject", ""),
        "task": row.get("task", ""),
        "raw_task": row.get("raw_task", ""),
        "run": row.get("run", ""),
        "release": row.get("release", ""),
        "raw_path": str(raw_path),
    }
    try:
        import mne

        raw = mne.io.read_raw_eeglab(raw_path, preload=False, verbose="ERROR")
        return {
            **base,
            "status": "ok",
            "n_channels": len(raw.ch_names),
            "sfreq": float(raw.info["sfreq"]),
            "n_times": int(raw.n_times),
            "duration_sec": float(raw.n_times / raw.info["sfreq"]),
            "first_channels": ",".join(raw.ch_names[:8]),
            "error": "",
        }
    except Exception as exc:
        return {
            **base,
            "status": "failed",
            "n_channels": None,
            "sfreq": None,
            "n_times": None,
            "duration_sec": None,
            "first_channels": "",
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-recordings", type=int, default=30)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.max_recordings > 0:
        # Preserve task diversity in a tiny preflight sample.
        manifest = (
            manifest.groupby("task", group_keys=False)
            .head(max(1, args.max_recordings // max(1, manifest["task"].nunique())))
            .head(args.max_recordings)
            .copy()
        )
    rows = [read_one(row) for _, row in manifest.iterrows()]
    out = pd.DataFrame(rows)
    ensure_parent(args.out)
    out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()


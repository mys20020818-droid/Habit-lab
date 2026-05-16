from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from s00_utils_common import ensure_parent, load_config


def derivative_path(config: dict, subject: str, task: str) -> Path:
    root = Path(config["paths"]["derivatives_root"])
    return root / subject / f"{subject}_task-{task}_preproc-epo.fif"


def preprocess_one(row: pd.Series, config: dict) -> dict:
    """Placeholder for the real MNE preprocessing step.

    Replace the stub block with:
    - mne.io.read_raw_* according to file extension
    - filter, notch_filter, interpolate_bads, set_eeg_reference
    - ICA fitting / component removal
    - epoching from task events
    - write_epochs(..., overwrite=True)
    """
    out_path = derivative_path(config, row["subject"], row["task"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return {
        "subject": row["subject"],
        "task": row["task"],
        "preprocessed_path": str(out_path),
        "qc_bad_epoch_fraction": pd.NA,
        "n_epochs_retained": pd.NA,
        "status": "stub_pending_real_preprocessing",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--marker", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = pd.read_csv(args.manifest)
    rows = [preprocess_one(row, config) for _, row in manifest.iterrows()]

    qc_path = Path(config["outputs"]["manifest"]).with_name("preprocessing_qc.csv")
    ensure_parent(qc_path)
    pd.DataFrame(rows).to_csv(qc_path, index=False)
    ensure_parent(args.marker)
    Path(args.marker).write_text("done\n", encoding="utf-8")


if __name__ == "__main__":
    main()


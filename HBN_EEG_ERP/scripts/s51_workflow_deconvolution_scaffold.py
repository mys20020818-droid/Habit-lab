from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from s00_utils_common import ensure_parent, load_config


def deconvolved_path(config: dict, subject: str, task: str, operation: str) -> Path:
    root = Path(config["paths"]["derivatives_root"])
    return root / subject / f"{subject}_task-{task}_operation-{operation}_deconv.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--marker", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    events = pd.read_csv(args.events)

    if not events.empty:
        for (subject, task, operation), chunk in events.groupby(["subject", "task", "operation"]):
            out_path = deconvolved_path(config, subject, task, operation)
            ensure_parent(out_path)
            # Replace this placeholder with Unfold/MNE regression output.
            pd.DataFrame(
                {
                    "subject": [subject],
                    "task": [task],
                    "operation": [operation],
                    "n_events": [len(chunk)],
                    "deconv_path": [str(out_path)],
                    "status": ["stub_pending_real_deconvolution"],
                }
            ).to_csv(out_path, index=False)

    ensure_parent(args.marker)
    Path(args.marker).write_text("done\n", encoding="utf-8")


if __name__ == "__main__":
    main()


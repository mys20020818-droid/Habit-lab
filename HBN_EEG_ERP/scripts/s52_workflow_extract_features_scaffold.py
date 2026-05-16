from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from s00_utils_common import ensure_parent, load_config


FEATURE_COLUMNS = [
    "subject",
    "task",
    "operation",
    "feature",
    "amplitude",
    "n_trials",
    "channel_group",
    "tmin",
    "tmax",
]


def deterministic_stub_value(subject: str, task: str, window: str) -> float:
    digest = hashlib.sha256(f"{subject}|{task}|{window}".encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    rng = np.random.default_rng(seed)
    return float(rng.normal(loc=0.0, scale=1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    events = pd.read_csv(args.events)
    feature_windows = config["features"]["windows"]
    rows: list[dict] = []

    if not events.empty:
        grouped = events.groupby(["subject", "task", "operation"], dropna=False)
        for (subject, task, operation), chunk in grouped:
            for window_name, spec in feature_windows.items():
                if spec["operation"] != operation:
                    continue
                rows.append(
                    {
                        "subject": subject,
                        "task": task,
                        "operation": operation,
                        "feature": window_name,
                        "amplitude": deterministic_stub_value(subject, task, window_name),
                        "n_trials": len(chunk),
                        "channel_group": spec["channel_group"],
                        "tmin": spec["tmin"],
                        "tmax": spec["tmax"],
                    }
                )

    features = pd.DataFrame(rows)
    ensure_parent(args.out)
    features.reindex(columns=FEATURE_COLUMNS).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

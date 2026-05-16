from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_parent(path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def normalize_subject(raw: str) -> str:
    value = str(raw)
    return value if value.startswith("sub-") else f"sub-{value}"


def normalize_task(raw: str) -> str:
    return str(raw).strip().replace("task-", "")


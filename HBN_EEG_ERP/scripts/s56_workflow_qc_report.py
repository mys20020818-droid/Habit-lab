from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from jinja2 import Template

from s00_utils_common import ensure_parent, load_config


TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>HBN-EEG QC Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }
    table { border-collapse: collapse; margin: 16px 0; min-width: 520px; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; }
    th { background: #f3f3f3; }
    code { background: #f6f6f6; padding: 2px 4px; }
  </style>
</head>
<body>
  <h1>HBN-EEG QC Report</h1>
  <p>Project: <code>{{ project_name }}</code></p>
  <h2>Tables</h2>
  <table>
    <tr><th>Table</th><th>Rows</th><th>Columns</th></tr>
    {% for row in tables %}
    <tr><td>{{ row.name }}</td><td>{{ row.rows }}</td><td>{{ row.cols }}</td></tr>
    {% endfor %}
  </table>
  <h2>Interpretation Gates</h2>
  <ul>
    <li>Do not interpret psychopathology associations before feature reliability is checked.</li>
    <li>Flag any feature with too few subjects, too few tasks, or failed LME/GAMM status.</li>
    <li>Use the pilot to decide whether the full mass-univariate model is justified.</li>
  </ul>
</body>
</html>
"""


def table_summary(name: str, path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"name": name, "rows": 0, "cols": 0}
    df = pd.read_csv(p)
    return {"name": name, "rows": len(df), "cols": len(df.columns)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--components", required=True)
    parser.add_argument("--developmental", required=True)
    parser.add_argument("--psychopathology", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    tables = [
        table_summary("manifest", args.manifest),
        table_summary("events", args.events),
        table_summary("features", args.features),
        table_summary("components", args.components),
        table_summary("developmental", args.developmental),
        table_summary("psychopathology", args.psychopathology),
    ]
    html = Template(TEMPLATE).render(
        project_name=config["project"]["name"],
        tables=tables,
    )
    ensure_parent(args.out)
    Path(args.out).write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()


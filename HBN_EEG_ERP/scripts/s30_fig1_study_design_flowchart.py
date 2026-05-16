from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures" / "manuscript_core_nature"
SOURCE_DIR = ROOT / "results" / "tables" / "manuscript_core_nature"
REPORT_PATH = ROOT / "results" / "reports" / "manuscript_figure1_flowchart_report.md"

BLUE = "#0F4D92"
BLUE_SOFT = "#DCE8F5"
RED = "#B64342"
RED_SOFT = "#F4DFDD"
TEAL = "#4FA79D"
TEAL_SOFT = "#E1F1EF"
GREEN = "#7BAA5B"
GREEN_SOFT = "#E7F1E0"
GOLD = "#D8A044"
GOLD_SOFT = "#F5EAD4"
TASK_COLORS = {"CCD": "#6F8DBF", "SuS": TEAL}
NEUTRAL_DARK = "#272727"
NEUTRAL_MID = "#767676"
NEUTRAL_LIGHT = "#D8D8D8"
NEUTRAL_FAINT = "#F5F5F5"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_pub(fig, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{base_path}.svg", bbox_inches="tight")
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base_path}.tiff", dpi=600, bbox_inches="tight")


def add_box(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    facecolor: str = NEUTRAL_FAINT,
    edgecolor: str = NEUTRAL_MID,
    linewidth: float = 0.75,
    fontsize: float = 7.2,
    weight: str = "normal",
    radius: float = 0.012,
    align: str = "center",
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(patch)
    ha = "center" if align == "center" else "left"
    tx = x + w / 2 if align == "center" else x + 0.018
    ax.text(
        tx,
        y + h / 2,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=NEUTRAL_DARK,
        linespacing=1.15,
        zorder=5,
    )
    return patch


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = NEUTRAL_MID, lw: float = 0.9) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=lw,
            color=color,
            transform=ax.transAxes,
            zorder=4,
            shrinkA=2,
            shrinkB=2,
        )
    )


def section_label(ax, label: str, title: str, x: float, y: float) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="top")
    ax.text(x + 0.026, y, title, transform=ax.transAxes, fontsize=8.4, fontweight="bold", ha="left", va="top")


def draw_stimulus(ax, center: tuple[float, float], scale: float, kind: str) -> None:
    cx, cy = center
    ax.add_patch(Rectangle((cx - 0.04 * scale, cy - 0.028 * scale), 0.08 * scale, 0.056 * scale, transform=ax.transAxes, facecolor="white", edgecolor=NEUTRAL_MID, linewidth=0.7, zorder=3))
    ax.add_patch(Rectangle((cx - 0.018 * scale, cy - 0.043 * scale), 0.036 * scale, 0.012 * scale, transform=ax.transAxes, facecolor=NEUTRAL_LIGHT, edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((cx - 0.032 * scale, cy - 0.053 * scale), 0.064 * scale, 0.007 * scale, transform=ax.transAxes, facecolor=NEUTRAL_LIGHT, edgecolor="none", zorder=2))
    if kind == "sus":
        ax.add_patch(Circle((cx, cy), 0.015 * scale, transform=ax.transAxes, facecolor=BLUE_SOFT, edgecolor=BLUE, linewidth=0.8, zorder=4))
        ax.add_patch(Circle((cx, cy), 0.027 * scale, transform=ax.transAxes, facecolor="none", edgecolor=BLUE, linewidth=1.2, zorder=4))
        ax.add_patch(Circle((cx, cy), 0.0028 * scale, transform=ax.transAxes, facecolor=NEUTRAL_DARK, edgecolor="none", zorder=5))
    else:
        ax.add_patch(Rectangle((cx - 0.026 * scale, cy - 0.017 * scale), 0.052 * scale, 0.034 * scale, transform=ax.transAxes, facecolor=GOLD_SOFT, edgecolor=GOLD, linewidth=0.8, zorder=4))
        for i in range(5):
            x0 = cx - 0.024 * scale + i * 0.012 * scale
            ax.plot([x0, x0 + 0.018 * scale], [cy - 0.016 * scale, cy + 0.016 * scale], transform=ax.transAxes, color=GOLD, linewidth=0.7, zorder=5)
        ax.add_patch(Circle((cx + 0.046 * scale, cy - 0.038 * scale), 0.009 * scale, transform=ax.transAxes, facecolor=RED, edgecolor="white", linewidth=0.4, zorder=5))


def draw_erp_icon(ax, x: float, y: float, w: float, h: float, color: str, label: str) -> None:
    xx = np.linspace(0, 1, 90)
    yy = (
        0.14 * np.sin(2 * np.pi * (xx * 2.2 + 0.08))
        - 0.42 * np.exp(-((xx - 0.34) / 0.08) ** 2)
        + 0.34 * np.exp(-((xx - 0.68) / 0.13) ** 2)
    )
    X = x + xx * w
    Y = y + h * (0.50 + yy)
    ax.plot([x, x + w], [y + h * 0.50, y + h * 0.50], transform=ax.transAxes, color=NEUTRAL_LIGHT, linewidth=0.55, zorder=2)
    ax.plot(X, Y, transform=ax.transAxes, color=color, linewidth=1.2, zorder=5)
    ax.text(x + w * 0.02, y + h * 0.90, label, transform=ax.transAxes, fontsize=6.4, color=color, ha="left", va="center")


def draw_trajectory_icon(ax, x: float, y: float, w: float, h: float) -> None:
    xx = np.linspace(0, 1, 100)
    y1 = 0.22 + 0.58 * xx + 0.06 * np.sin(xx * np.pi)
    y2 = 0.48 - 0.08 * xx + 0.04 * np.sin(xx * np.pi * 2)
    ax.fill_between(x + xx * w, y + (y1 - 0.07) * h, y + (y1 + 0.07) * h, transform=ax.transAxes, color=BLUE, alpha=0.10, linewidth=0)
    ax.plot(x + xx * w, y + y1 * h, transform=ax.transAxes, color=BLUE, linewidth=1.5)
    ax.plot(x + xx * w, y + y2 * h, transform=ax.transAxes, color=RED, linewidth=1.2)
    ax.plot([x, x], [y + 0.08 * h, y + 0.92 * h], transform=ax.transAxes, color=NEUTRAL_DARK, linewidth=0.65)
    ax.plot([x, x + w], [y + 0.08 * h, y + 0.08 * h], transform=ax.transAxes, color=NEUTRAL_DARK, linewidth=0.65)
    ax.text(x + w * 0.50, y - 0.015, "Age 5-21 years", transform=ax.transAxes, fontsize=6.3, ha="center", va="top")
    ax.text(x - 0.016, y + h * 0.50, "ERP score", transform=ax.transAxes, fontsize=6.3, ha="right", va="center", rotation=90)


def draw_figure() -> Path:
    set_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.6, 5.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column regions.
    ax.add_patch(Rectangle((0.025, 0.08), 0.275, 0.83, transform=ax.transAxes, facecolor="#FBFBFB", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.365, 0.08), 0.270, 0.83, transform=ax.transAxes, facecolor="#FBFBFB", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.700, 0.08), 0.275, 0.83, transform=ax.transAxes, facecolor="#FBFBFB", edgecolor="none", zorder=0))

    section_label(ax, "a", "Task paradigms", 0.035, 0.93)
    section_label(ax, "b", "Within-subject decomposition", 0.375, 0.93)
    section_label(ax, "c", "Developmental modeling", 0.710, 0.93)

    # Panel a: tasks.
    add_box(ax, (0.055, 0.665), (0.215, 0.145), "", TEAL_SOFT, TEAL, weight="bold")
    draw_stimulus(ax, (0.095, 0.735), 0.92, "sus")
    ax.text(0.145, 0.756, "Surround suppression", transform=ax.transAxes, fontsize=7.0, fontweight="bold", ha="left", va="center", color=NEUTRAL_DARK)
    ax.text(0.145, 0.710, "Passive viewing\nfixation + surround", transform=ax.transAxes, fontsize=6.2, ha="left", va="center", color=NEUTRAL_DARK, linespacing=1.15)
    add_box(ax, (0.055, 0.430), (0.215, 0.145), "", GOLD_SOFT, GOLD, weight="bold")
    draw_stimulus(ax, (0.095, 0.500), 0.92, "ccd")
    ax.text(0.145, 0.522, "Contrast change detection", transform=ax.transAxes, fontsize=7.0, fontweight="bold", ha="left", va="center", color=NEUTRAL_DARK)
    ax.text(0.145, 0.476, "Active decision\ncontrast change / response", transform=ax.transAxes, fontsize=6.2, ha="left", va="center", color=NEUTRAL_DARK, linespacing=1.15)
    add_box(ax, (0.070, 0.190), (0.185, 0.100), "Visual ERPs extracted\nP1, P2, N450 windows", "white", NEUTRAL_LIGHT, fontsize=6.7)
    add_arrow(ax, (0.160, 0.660), (0.160, 0.580), TEAL)
    add_arrow(ax, (0.160, 0.425), (0.160, 0.295), GOLD)

    # Panel b: decomposition.
    add_box(ax, (0.395, 0.665), (0.200, 0.145), "", "white", NEUTRAL_LIGHT, weight="bold")
    ax.text(0.495, 0.780, "Same participant", transform=ax.transAxes, fontsize=7.0, fontweight="bold", ha="center", va="center", color=NEUTRAL_DARK)
    ax.text(0.495, 0.754, "SuS and CCD ERPs", transform=ax.transAxes, fontsize=6.4, ha="center", va="center", color=NEUTRAL_DARK)
    draw_erp_icon(ax, 0.418, 0.690, 0.075, 0.055, TEAL, "SuS")
    draw_erp_icon(ax, 0.505, 0.690, 0.075, 0.055, TASK_COLORS["CCD"], "CCD")
    add_box(ax, (0.400, 0.438), (0.090, 0.108), "General\n(SuS + CCD)/2", BLUE_SOFT, BLUE, weight="bold")
    add_box(ax, (0.505, 0.438), (0.095, 0.108), "Specific\nSuS - CCD", RED_SOFT, RED, weight="bold")
    add_arrow(ax, (0.475, 0.660), (0.445, 0.550), BLUE)
    add_arrow(ax, (0.515, 0.660), (0.555, 0.550), RED)
    add_box(ax, (0.400, 0.210), (0.200, 0.105), "Variance and reliability checks\nICC + split-half stability", "white", NEUTRAL_LIGHT, fontsize=6.7)
    add_arrow(ax, (0.445, 0.435), (0.465, 0.320), BLUE)
    add_arrow(ax, (0.555, 0.435), (0.535, 0.320), RED)

    # Panel c: models.
    add_box(ax, (0.735, 0.682), (0.205, 0.120), "Age trajectory models", GREEN_SOFT, GREEN, weight="bold")
    draw_trajectory_icon(ax, 0.745, 0.465, 0.180, 0.155)
    add_box(ax, (0.732, 0.308), (0.095, 0.095), "General vs\nspecific shape", BLUE_SOFT, BLUE, fontsize=6.6, weight="bold")
    add_box(ax, (0.845, 0.308), (0.095, 0.095), "p-factor x age\nmoderation", RED_SOFT, RED, fontsize=6.6, weight="bold")
    add_box(ax, (0.748, 0.160), (0.175, 0.080), "Source-data outputs\nfigures + reproducible tables", "white", NEUTRAL_LIGHT, fontsize=6.6)
    add_arrow(ax, (0.837, 0.680), (0.837, 0.625), GREEN)
    add_arrow(ax, (0.810, 0.460), (0.785, 0.405), BLUE)
    add_arrow(ax, (0.875, 0.460), (0.895, 0.405), RED)
    add_arrow(ax, (0.785, 0.305), (0.810, 0.245), BLUE)
    add_arrow(ax, (0.895, 0.305), (0.865, 0.245), RED)

    # Cross-panel flow arrows.
    add_arrow(ax, (0.275, 0.500), (0.390, 0.720), NEUTRAL_MID)
    add_arrow(ax, (0.600, 0.490), (0.735, 0.742), NEUTRAL_MID)
    add_arrow(ax, (0.600, 0.260), (0.735, 0.355), NEUTRAL_MID)

    # Bottom evidence chain.
    ax.plot([0.070, 0.925], [0.070, 0.070], transform=ax.transAxes, color=NEUTRAL_LIGHT, linewidth=0.8)
    milestones = [
        (0.120, "SuS + CCD\nrecordings"),
        (0.330, "ERP features\nby subject"),
        (0.510, "General/specific\ncomponents"),
        (0.700, "P2/N450 age\ntrajectories"),
        (0.885, "Clinical null\ntransparency"),
    ]
    for x, text in milestones:
        ax.add_patch(Circle((x, 0.070), 0.008, transform=ax.transAxes, facecolor=NEUTRAL_DARK, edgecolor="white", linewidth=0.4, zorder=5))
        ax.text(x, 0.035, text, transform=ax.transAxes, ha="center", va="top", fontsize=5.9, color=NEUTRAL_DARK, linespacing=1.15)

    flow_nodes = pd.DataFrame(
        [
            {"panel": "a", "node": "Surround suppression", "role": "Passive visual ERP task"},
            {"panel": "a", "node": "Contrast change detection", "role": "Active visual ERP task"},
            {"panel": "b", "node": "General component", "role": "Cross-task shared ERP structure"},
            {"panel": "b", "node": "Specific component", "role": "Task-difference ERP structure"},
            {"panel": "c", "node": "Age models", "role": "Developmental trajectory estimation"},
            {"panel": "c", "node": "Clinical moderation", "role": "p-factor x age transparency analysis"},
        ]
    )
    flow_nodes.to_csv(SOURCE_DIR / "figure1_flowchart_nodes.csv", index=False)

    out = FIG_DIR / "fig1"
    save_pub(fig, out)
    plt.close(fig)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Figure 1 Flowchart",
                "",
                "Conceptual study-design schematic generated with Python/matplotlib as an editable vector draft.",
                "",
                "Panels:",
                "- a: SuS passive viewing and CCD active contrast-change task paradigms.",
                "- b: within-subject decomposition of two task ERPs into general and specific components.",
                "- c: age trajectory and clinical-moderation modeling workflow.",
                "",
                f"Output base: `{out.relative_to(ROOT)}`",
                f"Source nodes: `{(SOURCE_DIR / 'figure1_flowchart_nodes.csv').relative_to(ROOT)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return out


def main() -> None:
    out = draw_figure()
    print(f"saved: {out}")
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

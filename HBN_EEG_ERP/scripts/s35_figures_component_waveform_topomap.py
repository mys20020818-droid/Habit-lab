from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from s12_extract_selected_electrode_features import ERP_SELECTION, ROI_DEFS
from s34_figures_full_channel_topomaps import (
    AGE_COLORS,
    AGE_GROUPS,
    AGE_LABELS,
    COMPONENT_LABELS,
    COMPONENTS,
    load_cached_recordings,
    save_pub,
)


DEFAULT_ERPS = ["P1", "P2", "N450"]
CONTRASTS = [
    ("10-14 - 5-9", "adolescent_10_14", "child_5_9"),
    ("15-21 - 5-9", "youth_15_21", "child_5_9"),
]
DISPLAY_XLIM = (-0.20, 0.80)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )


def plot_topomap(data: np.ndarray, info, ax, vmax: float):
    import mne

    try:
        image, _ = mne.viz.plot_topomap(
            data,
            info,
            axes=ax,
            show=False,
            contours=0,
            cmap="RdBu_r",
            vlim=(-vmax, vmax),
            sensors=False,
            names=None,
        )
    except TypeError:
        image, _ = mne.viz.plot_topomap(
            data,
            info,
            axes=ax,
            show=False,
            contours=0,
            cmap="RdBu_r",
            sensors=False,
            names=None,
        )
        image.set_clim(-vmax, vmax)
    return image


def highlight_roi(ax, info, ch_names: list[str], channels: list[str]) -> None:
    from mne.channels.layout import _find_topomap_coords

    picks = [ch_names.index(ch) for ch in channels if ch in ch_names]
    if not picks:
        return
    coords = _find_topomap_coords(info, picks=picks)
    ax.scatter(coords[:, 0], coords[:, 1], s=7, facecolor="#111111", edgecolor="white", linewidth=0.25, zorder=20)


def channel_values(values: pd.DataFrame, component: str, erp: str, age_group: str, ch_names: list[str]) -> np.ndarray:
    sub = (
        values[
            values["component"].eq(component)
            & values["erp_component"].eq(erp)
            & values["age_group"].eq(age_group)
        ]
        .set_index("channel")
        .reindex(ch_names)
    )
    return sub["amplitude_uv"].to_numpy(dtype=float)


def contrast_values(values: pd.DataFrame, component: str, erp: str, age_a: str, age_b: str, ch_names: list[str]) -> np.ndarray:
    a = channel_values(values, component, erp, age_a, ch_names)
    b = channel_values(values, component, erp, age_b, ch_names)
    return a - b


def waveform(values: pd.DataFrame, component: str, erp: str, roi: str, age_group: str) -> pd.DataFrame:
    sub = values[
        values["component"].eq(component)
        & values["erp_component"].eq(erp)
        & values["roi"].eq(roi)
        & values["age_group"].eq(age_group)
    ].copy()
    return sub.sort_values("time_s")


def robust_ylim(waveforms: pd.DataFrame, component: str, erp: str, roi: str) -> tuple[float, float]:
    sub = waveforms[
        waveforms["component"].eq(component)
        & waveforms["erp_component"].eq(erp)
        & waveforms["roi"].eq(roi)
        & waveforms["time_s"].between(DISPLAY_XLIM[0], DISPLAY_XLIM[1])
    ]
    y = sub["amplitude_uv"].to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    lo, hi = np.percentile(y, [1, 99])
    pad = max(0.6, (hi - lo) * 0.12)
    return float(lo - pad), float(hi + pad)


def add_window_annotation(ax, erp: str, tmin: float, tmax: float, ylims: tuple[float, float]) -> None:
    ax.axvspan(tmin, tmax, color="#D9A441", alpha=0.13, linewidth=0, zorder=0)
    yr = ylims[1] - ylims[0]
    rect_y = ylims[0] + yr * 0.18
    rect_h = yr * 0.56
    rect = patches.Rectangle(
        (tmin, rect_y),
        tmax - tmin,
        rect_h,
        linewidth=0.7,
        edgecolor="#888888",
        facecolor="none",
        linestyle=(0, (1.5, 1.5)),
        alpha=0.70,
        zorder=1,
    )
    ax.add_patch(rect)
    ax.annotate(
        erp,
        xy=((tmin + tmax) / 2.0, rect_y + rect_h),
        xytext=(min(tmax + 0.13, DISPLAY_XLIM[1] - 0.04), ylims[1] - yr * 0.10),
        ha="center",
        va="center",
        fontsize=8,
        arrowprops=dict(arrowstyle="-", color="#444444", lw=0.7, shrinkA=0, shrinkB=0),
    )


def map_vmax(values: pd.DataFrame, erp: str) -> float:
    vals = []
    sub = values[values["erp_component"].eq(erp)]
    vals.extend(sub["amplitude_uv"].to_numpy(dtype=float))
    for component in COMPONENTS:
        for _label, age_a, age_b in CONTRASTS:
            a = sub[(sub["component"].eq(component)) & (sub["age_group"].eq(age_a))].set_index("channel")["amplitude_uv"]
            b = sub[(sub["component"].eq(component)) & (sub["age_group"].eq(age_b))].set_index("channel")["amplitude_uv"]
            common = a.index.intersection(b.index)
            vals.extend((a.loc[common] - b.loc[common]).to_numpy(dtype=float))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return max(0.5, float(np.nanpercentile(np.abs(vals), 98)))


def add_age_maps(
    fig,
    values: pd.DataFrame,
    info,
    ch_names: list[str],
    component: str,
    erp: str,
    channels: list[str],
    x0: float,
    y0: float,
    width: float,
    size: float,
    vmax: float,
) -> object:
    gap = width * 0.055
    total = size * 3 + gap * 2
    start = x0 + (width - total) / 2
    image = None
    for idx, age_group in enumerate(AGE_GROUPS):
        ax = fig.add_axes([start + idx * (size + gap), y0, size, size])
        data = channel_values(values, component, erp, age_group, ch_names)
        image = plot_topomap(data, info, ax, vmax)
        highlight_roi(ax, info, ch_names, channels)
        ax.text(0.5, -0.12, AGE_LABELS[age_group], transform=ax.transAxes, ha="center", va="top", fontsize=6)
    return image


def add_contrast_maps(
    fig,
    values: pd.DataFrame,
    info,
    ch_names: list[str],
    component: str,
    erp: str,
    channels: list[str],
    x0: float,
    y0: float,
    width: float,
    size: float,
    vmax: float,
) -> object:
    gap = width * 0.08
    total = size * len(CONTRASTS) + gap * (len(CONTRASTS) - 1)
    start = x0 + (width - total) / 2
    image = None
    for idx, (label, age_a, age_b) in enumerate(CONTRASTS):
        ax = fig.add_axes([start + idx * (size + gap), y0, size, size])
        data = contrast_values(values, component, erp, age_a, age_b, ch_names)
        image = plot_topomap(data, info, ax, vmax)
        highlight_roi(ax, info, ch_names, channels)
        ax.text(0.5, -0.12, label, transform=ax.transAxes, ha="center", va="top", fontsize=6)
    return image


def plot_component_figure(
    erp: str,
    waveforms: pd.DataFrame,
    values: pd.DataFrame,
    info,
    ch_names: list[str],
    out_dir: Path,
) -> None:
    spec = ERP_SELECTION[erp]
    roi = spec["roi"]
    tmin, tmax = spec["window"]
    channels = ROI_DEFS[roi]["channels"]
    vmax = map_vmax(values, erp)

    fig = plt.figure(figsize=(10.6, 5.4))
    panel_specs = [
        ("A", "general", 0.065, 0.10, 0.405, 0.80),
        ("B", "specific_sus_minus_ccd", 0.555, 0.10, 0.405, 0.80),
    ]
    last_image = None

    for panel, component, x0, y0, width, height in panel_specs:
        fig.text(x0 - 0.035, y0 + height + 0.045, panel, ha="left", va="top", fontsize=12, fontweight="bold")
        fig.text(
            x0 + width / 2,
            y0 + height + 0.025,
            f"{COMPONENT_LABELS[component]}",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
        )

        map_size = width * 0.17
        top_y = y0 + height * 0.75
        bottom_y = y0 + height * 0.035
        wave_y = y0 + height * 0.28
        wave_h = height * 0.39

        fig.text(
            x0 + width / 2,
            top_y + map_size + 0.022,
            f"{erp} topographies ({int(tmin * 1000)}-{int(tmax * 1000)} ms)",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
        )
        last_image = add_age_maps(fig, values, info, ch_names, component, erp, channels, x0, top_y, width, map_size, vmax)
        fig.add_artist(Line2D([x0, x0 + width], [top_y - 0.035, top_y - 0.035], transform=fig.transFigure, color="#666666", lw=0.6))

        ax = fig.add_axes([x0, wave_y, width, wave_h])
        ylims = robust_ylim(waveforms, component, erp, roi)
        for age_group in AGE_GROUPS:
            line = waveform(waveforms, component, erp, roi, age_group)
            ax.plot(
                line["time_s"],
                line["amplitude_uv"],
                color=AGE_COLORS[age_group],
                linewidth=1.20,
                label=AGE_LABELS[age_group],
            )
        ax.set_xlim(*DISPLAY_XLIM)
        ax.set_ylim(*ylims)
        ax.axvline(0, color="#666666", linewidth=0.9)
        ax.axhline(0, color="#BFBFBF", linewidth=0.7)
        add_window_annotation(ax, erp, tmin, tmax, ylims)
        ax.text(0.02, 0.92, roi, transform=ax.transAxes, ha="left", va="top", fontsize=7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Potential (uV)")
        ax.tick_params(labelsize=7, length=2)
        fig.add_artist(Line2D([x0, x0 + width], [bottom_y + map_size + 0.035, bottom_y + map_size + 0.035], transform=fig.transFigure, color="#666666", lw=0.6))
        fig.text(
            x0 + width / 2,
            bottom_y + map_size + 0.018,
            "Developmental contrasts",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
        )
        last_image = add_contrast_maps(fig, values, info, ch_names, component, erp, channels, x0, bottom_y, width, map_size, vmax)

    cax = fig.add_axes([0.975, 0.22, 0.012, 0.58])
    cbar = fig.colorbar(last_image, cax=cax)
    cbar.set_label("Amplitude / contrast (uV)", rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=6)

    fig.text(
        0.50,
        0.985,
        f"{erp} ROI waveforms and scalp topographies",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.50,
        0.955,
        f"Waveform ROI: {roi} ({', '.join(channels)})",
        ha="center",
        va="top",
        fontsize=7,
        color="#444444",
    )
    handles = [
        Line2D([0], [0], color=AGE_COLORS[age_group], lw=1.4, label=AGE_LABELS[age_group])
        for age_group in AGE_GROUPS
    ]
    fig.legend(
        handles=handles,
        title="Age",
        loc="upper center",
        bbox_to_anchor=(0.50, 0.905),
        ncol=3,
        frameon=False,
        fontsize=7,
        title_fontsize=7,
        handlelength=1.8,
        columnspacing=1.2,
    )

    save_pub(fig, out_dir / f"component_reference_style_{erp}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--waveforms", required=True)
    parser.add_argument("--window-values", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--components", default=",".join(DEFAULT_ERPS))
    args = parser.parse_args()

    configure_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    waveforms = pd.read_csv(args.waveforms)
    values = pd.read_csv(args.window_values)
    _subject_task, info, _times, ch_names, _recording_summary = load_cached_recordings(Path(args.cache_dir))

    erps = [item.strip() for item in args.components.split(",") if item.strip()]
    for erp in erps:
        if erp not in ERP_SELECTION:
            raise KeyError(f"Unknown ERP component: {erp}")
        plot_component_figure(erp, waveforms, values, info, ch_names, out_dir)
        spec = ERP_SELECTION[erp]
        print(f"{erp}: roi={spec['roi']} window={spec['window'][0]:.2f}-{spec['window'][1]:.2f}s")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()

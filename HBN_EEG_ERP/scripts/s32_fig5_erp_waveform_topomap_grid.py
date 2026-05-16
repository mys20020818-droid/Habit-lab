from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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


ERPS = ["P1", "N1", "P3"]
ERP_DISPLAY_LABELS = {"P1": "P1", "N1": "P2", "P3": "N450"}
DISPLAY_XLIM = (-0.20, 0.80)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.75,
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
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=6,
        facecolor="#111111",
        edgecolor="white",
        linewidth=0.25,
        zorder=20,
    )


def waveform(waveforms: pd.DataFrame, component: str, erp: str, roi: str, age_group: str) -> pd.DataFrame:
    sub = waveforms[
        waveforms["component"].eq(component)
        & waveforms["erp_component"].eq(erp)
        & waveforms["roi"].eq(roi)
        & waveforms["age_group"].eq(age_group)
    ].copy()
    return sub.sort_values("time_s")


def topomap_values(values: pd.DataFrame, component: str, erp: str, age_group: str, ch_names: list[str]) -> np.ndarray:
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


def waveform_ylim(waveforms: pd.DataFrame, component: str, erp: str, roi: str) -> tuple[float, float]:
    sub = waveforms[
        waveforms["component"].eq(component)
        & waveforms["erp_component"].eq(erp)
        & waveforms["roi"].eq(roi)
        & waveforms["time_s"].between(DISPLAY_XLIM[0], DISPLAY_XLIM[1])
    ]
    y = sub["amplitude_uv"].to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return -1.0, 1.0
    lo, hi = np.nanpercentile(y, [1, 99])
    pad = max(0.45, (hi - lo) * 0.12)
    return float(lo - pad), float(hi + pad)


def topomap_vmax(values: pd.DataFrame) -> float:
    vals = values[values["erp_component"].isin(ERPS)]["amplitude_uv"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return max(0.5, float(np.nanpercentile(np.abs(vals), 98)))


def add_topomap_triplet(
    fig,
    container_ax,
    values: pd.DataFrame,
    info,
    ch_names: list[str],
    component: str,
    erp: str,
    channels: list[str],
    vmax: float,
    show_age_labels: bool,
):
    bbox = container_ax.get_position()
    container_ax.axis("off")
    gap = bbox.width * 0.06
    size = min(bbox.height * 0.86, (bbox.width - 2 * gap) / 3.0)
    total = 3 * size + 2 * gap
    x0 = bbox.x0 + (bbox.width - total) / 2.0
    y0 = bbox.y0 + (bbox.height - size) / 2.0
    image = None
    for idx, age_group in enumerate(AGE_GROUPS):
        ax = fig.add_axes([x0 + idx * (size + gap), y0, size, size])
        data = topomap_values(values, component, erp, age_group, ch_names)
        image = plot_topomap(data, info, ax, vmax)
        highlight_roi(ax, info, ch_names, channels)
        if show_age_labels:
            ax.text(
                0.5,
                -0.10,
                AGE_LABELS[age_group],
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=5.4,
            )
    return image


def plot_waveform_panel(ax, waveforms: pd.DataFrame, component: str, erp: str, roi: str) -> None:
    tmin, tmax = ERP_SELECTION[erp]["window"]
    ylims = waveform_ylim(waveforms, component, erp, roi)
    ax.axvspan(tmin, tmax, color="#D9A441", alpha=0.15, linewidth=0, zorder=0)
    for age_group in AGE_GROUPS:
        line = waveform(waveforms, component, erp, roi, age_group)
        ax.plot(
            line["time_s"],
            line["amplitude_uv"],
            color=AGE_COLORS[age_group],
            linewidth=1.05,
            label=AGE_LABELS[age_group],
        )
    ax.set_xlim(*DISPLAY_XLIM)
    ax.set_ylim(*ylims)
    ax.axvline(0, color="#777777", linewidth=0.75)
    ax.axhline(0, color="#D5D5D5", linewidth=0.65)
    ax.tick_params(labelsize=5.8, length=2)


def make_figure(waveforms: pd.DataFrame, values: pd.DataFrame, info, ch_names: list[str], out_dir: Path) -> None:
    configure_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    vmax = topomap_vmax(values)

    ncols = len(ERPS)
    fig = plt.figure(figsize=(9.4, 6.9))
    gs = fig.add_gridspec(
        nrows=4,
        ncols=ncols,
        left=0.065,
        right=0.925,
        top=0.835,
        bottom=0.105,
        height_ratios=[0.58, 1.28, 0.58, 1.28],
        wspace=0.27,
        hspace=0.32,
    )

    last_image = None
    for col, erp in enumerate(ERPS):
        spec = ERP_SELECTION[erp]
        roi = spec["roi"]
        tmin, tmax = spec["window"]
        channels = ROI_DEFS[roi]["channels"]
        fig.text(
            0.065 + col * ((0.925 - 0.065) / ncols) + ((0.925 - 0.065) / (2.0 * ncols)),
            0.878,
            f"{ERP_DISPLAY_LABELS[erp]} ({int(tmin * 1000)}-{int(tmax * 1000)} ms)\n{roi}",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
        )

        for row_block, component in enumerate(COMPONENTS):
            topo_row = row_block * 2
            wave_row = topo_row + 1
            ax_topo_container = fig.add_subplot(gs[topo_row, col])
            last_image = add_topomap_triplet(
                fig,
                ax_topo_container,
                values,
                info,
                ch_names,
                component,
                erp,
                channels,
                vmax,
                show_age_labels=True,
            )

            ax_wave = fig.add_subplot(gs[wave_row, col])
            plot_waveform_panel(ax_wave, waveforms, component, erp, roi)
            if row_block == 1:
                ax_wave.set_xlabel("Time (s)", fontsize=6.5)
            else:
                ax_wave.set_xticklabels([])
            if col == 0:
                ax_wave.set_ylabel("Potential (uV)", fontsize=6.5)
            else:
                ax_wave.set_yticklabels([])
            if col == ncols - 1 and row_block == 0:
                handles = [
                    Line2D([0], [0], color=AGE_COLORS[age_group], lw=1.3, label=AGE_LABELS[age_group])
                    for age_group in AGE_GROUPS
                ]
                ax_wave.legend(
                    handles=handles,
                    title="Age",
                    frameon=False,
                    fontsize=6,
                    title_fontsize=6,
                    loc="upper right",
                    handlelength=1.8,
                )

    fig.text(0.020, 0.675, COMPONENT_LABELS["general"], rotation=90, ha="center", va="center", fontsize=8, fontweight="bold")
    fig.text(
        0.020,
        0.285,
        COMPONENT_LABELS["specific_sus_minus_ccd"],
        rotation=90,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )
    fig.text(
        0.50,
        0.982,
        "ERP waveforms and age-group scalp topographies",
        ha="center",
        va="top",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.50,
        0.946,
        "Small topographies show age-group mean amplitudes; dots mark the waveform ROI electrodes.",
        ha="center",
        va="top",
        fontsize=7,
        color="#444444",
    )

    cax = fig.add_axes([0.945, 0.26, 0.012, 0.46])
    cbar = fig.colorbar(last_image, cax=cax)
    cbar.set_label("Topomap amplitude (uV)", rotation=270, labelpad=9)
    cbar.ax.tick_params(labelsize=5.5)

    save_pub(fig, out_dir / "fig5")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--waveforms", required=True)
    parser.add_argument("--window-values", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    waveforms = pd.read_csv(args.waveforms)
    values = pd.read_csv(args.window_values)
    if "erp_component" not in values.columns and "feature" in values.columns:
        values = values.rename(columns={"feature": "erp_component"})
    _, info, _, ch_names, _ = load_cached_recordings(Path(args.cache_dir))
    make_figure(waveforms, values, info, ch_names, Path(args.out_dir))
    print(f"saved: {args.out_dir}")


if __name__ == "__main__":
    main()

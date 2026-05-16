from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from s00_utils_common import ensure_parent
from s34_figures_full_channel_topomaps import (
    AGE_GROUPS,
    AGE_LABELS,
    COMPONENT_LABELS,
    COMPONENTS,
    build_components,
    group_means,
    load_cached_recordings,
    save_pub,
)


ERP_WINDOWS = {
    "P1": (0.08, 0.14),
    "P2": (0.14, 0.22),
    "N450": (0.30, 0.60),
}


def window_mean(evoked: np.ndarray, times: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    mask = (times >= tmin) & (times <= tmax)
    if not np.any(mask):
        raise ValueError(f"No time points in window {tmin}-{tmax}")
    return evoked[:, mask].mean(axis=1)


def build_extended_tables(means: dict, times: np.ndarray, ch_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    value_rows = []
    contrast_rows = []
    for (component, age_group), evoked in means.items():
        for erp_component, (tmin, tmax) in ERP_WINDOWS.items():
            values = window_mean(evoked, times, tmin, tmax)
            for ch_name, amp in zip(ch_names, values):
                value_rows.append(
                    {
                        "component": component,
                        "age_group": age_group,
                        "erp_component": erp_component,
                        "tmin_s": tmin,
                        "tmax_s": tmax,
                        "channel": ch_name,
                        "amplitude_uv": float(amp),
                    }
                )
    values = pd.DataFrame(value_rows)
    for component in COMPONENTS:
        for erp_component in ERP_WINDOWS:
            child = values[
                values["component"].eq(component)
                & values["age_group"].eq("child_5_9")
                & values["erp_component"].eq(erp_component)
            ].set_index("channel")["amplitude_uv"]
            youth = values[
                values["component"].eq(component)
                & values["age_group"].eq("youth_15_21")
                & values["erp_component"].eq(erp_component)
            ].set_index("channel")["amplitude_uv"]
            for ch_name in child.index.intersection(youth.index):
                contrast_rows.append(
                    {
                        "component": component,
                        "contrast": "youth_15_21_minus_child_5_9",
                        "erp_component": erp_component,
                        "tmin_s": ERP_WINDOWS[erp_component][0],
                        "tmax_s": ERP_WINDOWS[erp_component][1],
                        "channel": ch_name,
                        "delta_uv": float(youth.loc[ch_name] - child.loc[ch_name]),
                    }
                )
    contrasts = pd.DataFrame(contrast_rows)
    scale_rows = []
    for erp_component in ERP_WINDOWS:
        vals = contrasts[contrasts["erp_component"].eq(erp_component)]["delta_uv"].to_numpy(dtype=float)
        scale_rows.append(
            {
                "figure": "age_contrast",
                "erp_component": erp_component,
                "vmax_uv": max(0.1, float(np.nanpercentile(np.abs(vals), 98))),
                "scaling": "symmetric per-window 98th percentile across general and specific contrasts",
            }
        )
    for erp_component in ERP_WINDOWS:
        vals = values[values["erp_component"].eq(erp_component)]["amplitude_uv"].to_numpy(dtype=float)
        scale_rows.append(
            {
                "figure": "age_group_mean",
                "erp_component": erp_component,
                "vmax_uv": max(0.1, float(np.nanpercentile(np.abs(vals), 98))),
                "scaling": "symmetric per-window 98th percentile across components and age groups",
            }
        )
    return values, contrasts, pd.DataFrame(scale_rows)


def _plot_one_topomap(data: np.ndarray, info, ax, vmax: float):
    import mne

    try:
        im, _ = mne.viz.plot_topomap(
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
        im, _ = mne.viz.plot_topomap(
            data,
            info,
            axes=ax,
            show=False,
            contours=0,
            cmap="RdBu_r",
            sensors=False,
            names=None,
        )
        im.set_clim(-vmax, vmax)
    return im


def plot_contrast_grid(contrasts: pd.DataFrame, scales: pd.DataFrame, info, ch_names: list[str], out_base: Path) -> None:
    fig, axes = plt.subplots(2, len(ERP_WINDOWS), figsize=(12.0, 3.7))
    last_im = None
    scale_map = scales[scales["figure"].eq("age_contrast")].set_index("erp_component")["vmax_uv"].to_dict()
    for r, component in enumerate(COMPONENTS):
        for c, erp_component in enumerate(ERP_WINDOWS):
            ax = axes[r, c]
            row = contrasts[
                contrasts["component"].eq(component)
                & contrasts["erp_component"].eq(erp_component)
            ].set_index("channel").reindex(ch_names)
            vmax = scale_map[erp_component]
            last_im = _plot_one_topomap(row["delta_uv"].to_numpy(dtype=float), info, ax, vmax)
            tmin, tmax = ERP_WINDOWS[erp_component]
            if r == 0:
                ax.set_title(f"{erp_component}\n{int(tmin * 1000)}-{int(tmax * 1000)} ms", fontsize=7, pad=2)
            ax.text(
                0.5,
                -0.07,
                "General" if component == "general" else "Specific",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=6,
            )
    fig.subplots_adjust(left=0.02, right=0.93, bottom=0.08, top=0.82, wspace=0.04, hspace=0.22)
    cax = fig.add_axes([0.945, 0.18, 0.015, 0.56])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("Delta (uV)", rotation=270, labelpad=10)
    fig.suptitle("Extended full-channel ERP topographies: age contrast 15-21 minus 5-9", y=0.96, fontsize=9, fontweight="bold")
    save_pub(fig, out_base)
    plt.close(fig)


def plot_age_group_grid(values: pd.DataFrame, scales: pd.DataFrame, info, ch_names: list[str], out_base: Path) -> None:
    row_keys = [(component, age_group) for component in COMPONENTS for age_group in AGE_GROUPS]
    fig, axes = plt.subplots(len(row_keys), len(ERP_WINDOWS), figsize=(12.0, 8.4))
    last_im = None
    scale_map = scales[scales["figure"].eq("age_group_mean")].set_index("erp_component")["vmax_uv"].to_dict()
    for r, (component, age_group) in enumerate(row_keys):
        for c, erp_component in enumerate(ERP_WINDOWS):
            ax = axes[r, c]
            row = values[
                values["component"].eq(component)
                & values["age_group"].eq(age_group)
                & values["erp_component"].eq(erp_component)
            ].set_index("channel").reindex(ch_names)
            last_im = _plot_one_topomap(row["amplitude_uv"].to_numpy(dtype=float), info, ax, scale_map[erp_component])
            if r == 0:
                tmin, tmax = ERP_WINDOWS[erp_component]
                ax.set_title(f"{erp_component}\n{int(tmin * 1000)}-{int(tmax * 1000)} ms", fontsize=7, pad=2)
            if c == 0:
                label = "General" if component == "general" else "Specific"
                ax.text(-0.10, 0.5, f"{label}\n{AGE_LABELS[age_group]}", transform=ax.transAxes, ha="right", va="center", fontsize=6)
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.03, top=0.90, wspace=0.04, hspace=0.12)
    cax = fig.add_axes([0.945, 0.20, 0.015, 0.58])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("Amplitude (uV)", rotation=270, labelpad=10)
    fig.suptitle("Extended full-channel ERP topographies: age-group means", y=0.985, fontsize=9, fontweight="bold")
    save_pub(fig, out_base)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    subject_task, info, times, ch_names, recording_summary = load_cached_recordings(Path(args.cache_dir))
    components = build_components(subject_task)
    means, sample_summary = group_means(components)
    values, contrasts, scales = build_extended_tables(means, times, ch_names)

    values.to_csv(out_dir / "extended_full_channel_erp_window_values.csv", index=False)
    contrasts.to_csv(out_dir / "extended_full_channel_erp_age_contrasts.csv", index=False)
    scales.to_csv(out_dir / "extended_full_channel_erp_topomap_scales.csv", index=False)
    sample_summary.to_csv(out_dir / "extended_full_channel_erp_sample_summary.csv", index=False)
    recording_summary.to_csv(out_dir / "extended_full_channel_cached_recording_summary.csv", index=False)

    plot_contrast_grid(
        contrasts,
        scales,
        info,
        ch_names,
        figures_dir / "extended_full_channel_erp_topomaps_age_contrasts",
    )
    plot_age_group_grid(
        values,
        scales,
        info,
        ch_names,
        figures_dir / "extended_full_channel_erp_topomaps_age_group_means",
    )
    print(sample_summary.to_string(index=False))
    print(f"windows: {', '.join(ERP_WINDOWS)}")
    print(f"channels: {len(ch_names)}")
    print(f"cached recordings used: {len(recording_summary)}")


if __name__ == "__main__":
    main()

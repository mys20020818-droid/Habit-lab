from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from s34_figures_full_channel_topomaps import (
    AGE_COLORS,
    AGE_GROUPS,
    AGE_LABELS,
    COMPONENT_LABELS,
    COMPONENTS,
    build_components,
    load_cached_recordings,
    save_pub,
)


ROI_DEFS = {
    "occipital_core": {
        "channels": ["E70", "E71", "E72", "E75", "E76", "E83"],
        "rationale": "medial posterior/occipital channels for the earliest visual response",
    },
    "visual_posterior": {
        "channels": ["E65", "E66", "E70", "E71", "E75", "E76", "E83", "E84"],
        "rationale": "a priori visual posterior ROI used in the main P1/P2 analysis",
    },
    "frontocentral": {
        "channels": ["E5", "E6", "E7", "E12", "E13", "E106", "E112"],
        "rationale": "a priori frontocentral ROI used in the main N450 analysis",
    },
}


ERP_SELECTION = {
    "C1": {"window": (0.05, 0.09), "roi": "occipital_core"},
    "P1": {"window": (0.08, 0.14), "roi": "visual_posterior"},
    "N1": {"window": (0.14, 0.22), "roi": "visual_posterior"},
    "P2": {"window": (0.22, 0.30), "roi": "visual_posterior"},
    "N2": {"window": (0.25, 0.35), "roi": "frontocentral"},
    "N450": {"window": (0.30, 0.60), "roi": "frontocentral"},
}


def channel_indices(ch_names: list[str], channels: list[str]) -> list[int]:
    missing = [ch for ch in channels if ch not in ch_names]
    if missing:
        raise ValueError(f"Missing selected channels: {missing}")
    return [ch_names.index(ch) for ch in channels]


def selected_roi_table(ch_names: list[str]) -> pd.DataFrame:
    rows = []
    for roi_name, roi in ROI_DEFS.items():
        for channel in roi["channels"]:
            rows.append(
                {
                    "roi": roi_name,
                    "channel": channel,
                    "channel_index": ch_names.index(channel) if channel in ch_names else np.nan,
                    "rationale": roi["rationale"],
                }
            )
    return pd.DataFrame(rows)


def selected_component_table() -> pd.DataFrame:
    rows = []
    for erp_component, spec in ERP_SELECTION.items():
        tmin, tmax = spec["window"]
        rows.append(
            {
                "erp_component": erp_component,
                "roi": spec["roi"],
                "tmin_s": tmin,
                "tmax_s": tmax,
                "channels": ",".join(ROI_DEFS[spec["roi"]]["channels"]),
                "rationale": ROI_DEFS[spec["roi"]]["rationale"],
            }
        )
    return pd.DataFrame(rows)


def build_selected_features(components: dict, times: np.ndarray, ch_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows = []
    waveform_rows = []
    roi_indices = {roi: channel_indices(ch_names, spec["channels"]) for roi, spec in ROI_DEFS.items()}
    for component in COMPONENTS:
        for entry in components[component]:
            evoked = entry["evoked"]
            for erp_component, spec in ERP_SELECTION.items():
                roi = spec["roi"]
                picks = roi_indices[roi]
                tmin, tmax = spec["window"]
                mask = (times >= tmin) & (times <= tmax)
                amp = evoked[picks][:, mask].mean()
                feature_rows.append(
                    {
                        "subject": entry["subject"],
                        "age_group": entry["age_group"],
                        "component": component,
                        "erp_component": erp_component,
                        "roi": roi,
                        "channels": ",".join(ROI_DEFS[roi]["channels"]),
                        "tmin_s": tmin,
                        "tmax_s": tmax,
                        "amplitude_uv": float(amp),
                    }
                )
                wave = evoked[picks].mean(axis=0)
                for t, value in zip(times, wave):
                    waveform_rows.append(
                        {
                            "subject": entry["subject"],
                            "age_group": entry["age_group"],
                            "component": component,
                            "erp_component": erp_component,
                            "roi": roi,
                            "time_s": float(t),
                            "amplitude_uv": float(value),
                        }
                    )
    return pd.DataFrame(feature_rows), pd.DataFrame(waveform_rows)


def summarize_features(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = features.groupby(["component", "erp_component", "roi", "age_group"], sort=False)
    for keys, sub in grouped:
        component, erp_component, roi, age_group = keys
        rows.append(
            {
                "component": component,
                "erp_component": erp_component,
                "roi": roi,
                "age_group": age_group,
                "n_subjects": int(sub["subject"].nunique()),
                "mean_uv": float(sub["amplitude_uv"].mean()),
                "sd_uv": float(sub["amplitude_uv"].std(ddof=1)),
                "se_uv": float(sub["amplitude_uv"].std(ddof=1) / np.sqrt(sub["subject"].nunique())),
            }
        )
    return pd.DataFrame(rows)


def group_waveforms(waveforms: pd.DataFrame) -> pd.DataFrame:
    return (
        waveforms.groupby(["component", "erp_component", "roi", "age_group", "time_s"], sort=False)["amplitude_uv"]
        .mean()
        .reset_index()
    )


def plot_selected_waveforms(grouped_waveforms: pd.DataFrame, out_base: Path) -> None:
    fig, axes = plt.subplots(2, len(ERP_SELECTION), figsize=(12.2, 4.4), sharex=True)
    for r, component in enumerate(COMPONENTS):
        for c, erp_component in enumerate(ERP_SELECTION):
            spec = ERP_SELECTION[erp_component]
            tmin, tmax = spec["window"]
            roi = spec["roi"]
            ax = axes[r, c]
            sub = grouped_waveforms[
                grouped_waveforms["component"].eq(component)
                & grouped_waveforms["erp_component"].eq(erp_component)
                & grouped_waveforms["roi"].eq(roi)
            ]
            for age_group in AGE_GROUPS:
                line = sub[sub["age_group"].eq(age_group)]
                ax.plot(line["time_s"], line["amplitude_uv"], color=AGE_COLORS[age_group], linewidth=1.1, label=AGE_LABELS[age_group])
            ax.axvline(0, color="#777777", linewidth=0.7)
            ax.axhline(0, color="#D8D8D8", linewidth=0.7)
            ax.axvspan(tmin, tmax, color="#D9A441", alpha=0.12, linewidth=0)
            ax.set_xlim(-0.05, 0.75)
            if r == 0:
                ax.set_title(f"{erp_component}\n{roi}", fontsize=6.5)
            if c == 0:
                ax.set_ylabel(f"{COMPONENT_LABELS[component]}\nAmplitude (uV)")
            if r == 1:
                ax.set_xlabel("Time (s)")
            if r == 0 and c == len(ERP_SELECTION) - 1:
                ax.legend(frameon=False, fontsize=6, title="Age")
    fig.suptitle("Selected-electrode ERP waveforms", y=0.99, fontsize=9, fontweight="bold")
    fig.tight_layout(h_pad=1.0, w_pad=0.7)
    save_pub(fig, out_base)
    plt.close(fig)


def plot_selected_amplitudes(summary: pd.DataFrame, out_base: Path) -> None:
    x = np.arange(len(ERP_SELECTION))
    width = 0.24
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.8), sharex=True)
    for r, component in enumerate(COMPONENTS):
        ax = axes[r]
        sub = summary[summary["component"].eq(component)]
        for offset, age_group in zip([-width, 0, width], AGE_GROUPS):
            vals = []
            errs = []
            for erp_component in ERP_SELECTION:
                row = sub[(sub["erp_component"].eq(erp_component)) & (sub["age_group"].eq(age_group))]
                vals.append(float(row["mean_uv"].iloc[0]))
                errs.append(float(row["se_uv"].iloc[0]))
            ax.errorbar(
                x + offset,
                vals,
                yerr=errs,
                fmt="o",
                color=AGE_COLORS[age_group],
                markersize=3.8,
                linewidth=0.9,
                capsize=2,
                label=AGE_LABELS[age_group],
            )
        ax.axhline(0, color="#D8D8D8", linewidth=0.8)
        ax.set_ylabel(f"{COMPONENT_LABELS[component]}\nAmplitude (uV)")
        ax.set_title(COMPONENT_LABELS[component], fontsize=8)
        if r == 0:
            ax.legend(frameon=False, ncol=3, fontsize=6, title="Age")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(list(ERP_SELECTION))
    axes[-1].set_xlabel("ERP component")
    fig.suptitle("Selected-electrode window amplitudes", y=0.99, fontsize=9, fontweight="bold")
    fig.tight_layout(h_pad=1.0)
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

    subject_task, _info, times, ch_names, recording_summary = load_cached_recordings(Path(args.cache_dir))
    components = build_components(subject_task)

    rois = selected_roi_table(ch_names)
    component_selection = selected_component_table()
    features, waveforms = build_selected_features(components, times, ch_names)
    summary = summarize_features(features)
    grouped_waveforms = group_waveforms(waveforms)

    rois.to_csv(out_dir / "selected_electrode_rois.csv", index=False)
    component_selection.to_csv(out_dir / "selected_erp_component_electrodes.csv", index=False)
    features.to_csv(out_dir / "selected_electrode_erp_features_by_subject.csv", index=False)
    summary.to_csv(out_dir / "selected_electrode_erp_group_summary.csv", index=False)
    grouped_waveforms.to_csv(out_dir / "selected_electrode_erp_group_waveforms.csv", index=False)
    recording_summary.to_csv(out_dir / "selected_electrode_cached_recording_summary.csv", index=False)

    plot_selected_waveforms(grouped_waveforms, figures_dir / "selected_electrode_erp_waveforms")
    plot_selected_amplitudes(summary, figures_dir / "selected_electrode_erp_window_amplitudes")

    print(component_selection.to_string(index=False))
    print(summary.head(12).to_string(index=False))
    print(f"subjects: {features['subject'].nunique()}")


if __name__ == "__main__":
    main()

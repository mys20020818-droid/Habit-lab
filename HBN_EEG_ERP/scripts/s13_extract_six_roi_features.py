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


SIX_ROIS = {
    "left_frontal": ["E18", "E19", "E20", "E22", "E23", "E24", "E25", "E26"],
    "right_frontal": ["E2", "E3", "E4", "E8", "E9", "E10", "E118", "E124"],
    "frontocentral_midline": ["E5", "E6", "E7", "E11", "E12", "E13", "E15", "E16", "E17", "E106", "E112", "Cz"],
    "left_posterior": ["E29", "E30", "E31", "E36", "E37", "E42", "E53", "E54", "E60", "E61", "E65", "E66", "E67"],
    "right_posterior": ["E77", "E78", "E79", "E80", "E84", "E85", "E86", "E87", "E90", "E93", "E104", "E105", "E111"],
    "midline_occipital": ["E62", "E70", "E71", "E72", "E75", "E76", "E83"],
}


ERP_WINDOWS = {
    "P1": (0.08, 0.14),
    "P2": (0.14, 0.22),
    "N450": (0.30, 0.60),
}


def channel_indices(ch_names: list[str], channels: list[str]) -> list[int]:
    missing = [ch for ch in channels if ch not in ch_names]
    if missing:
        raise ValueError(f"Missing ROI channels: {missing}")
    return [ch_names.index(ch) for ch in channels]


def roi_definition_table(ch_names: list[str]) -> pd.DataFrame:
    rows = []
    for roi, channels in SIX_ROIS.items():
        for channel in channels:
            rows.append(
                {
                    "roi": roi,
                    "channel": channel,
                    "channel_index": ch_names.index(channel),
                    "n_channels_in_roi": len(channels),
                }
            )
    return pd.DataFrame(rows)


def build_features_and_group_waveforms(
    components: dict,
    times: np.ndarray,
    ch_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    roi_picks = {roi: channel_indices(ch_names, channels) for roi, channels in SIX_ROIS.items()}
    window_masks = {name: (times >= tmin) & (times <= tmax) for name, (tmin, tmax) in ERP_WINDOWS.items()}

    feature_rows = []
    wave_sums: dict[tuple[str, str, str], np.ndarray] = {}
    wave_counts: dict[tuple[str, str, str], int] = {}

    for component in COMPONENTS:
        for entry in components[component]:
            subject = entry["subject"]
            age_group = entry["age_group"]
            evoked = entry["evoked"]
            for roi, picks in roi_picks.items():
                roi_wave = evoked[picks].mean(axis=0)
                wave_key = (component, roi, age_group)
                if wave_key not in wave_sums:
                    wave_sums[wave_key] = np.zeros_like(roi_wave, dtype=float)
                    wave_counts[wave_key] = 0
                wave_sums[wave_key] += roi_wave
                wave_counts[wave_key] += 1
                for erp_component, mask in window_masks.items():
                    tmin, tmax = ERP_WINDOWS[erp_component]
                    feature_rows.append(
                        {
                            "subject": subject,
                            "age_group": age_group,
                            "component": component,
                            "erp_component": erp_component,
                            "roi": roi,
                            "n_channels": len(picks),
                            "channels": ",".join(SIX_ROIS[roi]),
                            "tmin_s": tmin,
                            "tmax_s": tmax,
                            "amplitude_uv": float(roi_wave[mask].mean()),
                        }
                    )

    wave_rows = []
    for (component, roi, age_group), summed in wave_sums.items():
        mean_wave = summed / wave_counts[(component, roi, age_group)]
        for time, amplitude in zip(times, mean_wave):
            wave_rows.append(
                {
                    "component": component,
                    "roi": roi,
                    "age_group": age_group,
                    "n_subjects": wave_counts[(component, roi, age_group)],
                    "time_s": float(time),
                    "amplitude_uv": float(amplitude),
                }
            )
    return pd.DataFrame(feature_rows), pd.DataFrame(wave_rows)


def summarize_features(features: pd.DataFrame) -> pd.DataFrame:
    grouped = features.groupby(["component", "erp_component", "roi", "age_group"], sort=False)
    rows = []
    for keys, sub in grouped:
        component, erp_component, roi, age_group = keys
        n = sub["subject"].nunique()
        sd = sub["amplitude_uv"].std(ddof=1)
        rows.append(
            {
                "component": component,
                "erp_component": erp_component,
                "roi": roi,
                "age_group": age_group,
                "n_subjects": int(n),
                "mean_uv": float(sub["amplitude_uv"].mean()),
                "sd_uv": float(sd),
                "se_uv": float(sd / np.sqrt(n)),
            }
        )
    return pd.DataFrame(rows)


def plot_roi_waveforms(group_waveforms: pd.DataFrame, out_base: Path) -> None:
    rois = list(SIX_ROIS)
    fig, axes = plt.subplots(2, len(rois), figsize=(12.0, 4.8), sharex=True)
    for r, component in enumerate(COMPONENTS):
        for c, roi in enumerate(rois):
            ax = axes[r, c]
            sub = group_waveforms[group_waveforms["component"].eq(component) & group_waveforms["roi"].eq(roi)]
            for age_group in AGE_GROUPS:
                line = sub[sub["age_group"].eq(age_group)]
                ax.plot(line["time_s"], line["amplitude_uv"], color=AGE_COLORS[age_group], linewidth=1.1, label=AGE_LABELS[age_group])
            ax.axvline(0, color="#777777", linewidth=0.7)
            ax.axhline(0, color="#D8D8D8", linewidth=0.7)
            ax.axvspan(0.08, 0.14, color="#8FB8DE", alpha=0.10, linewidth=0)
            ax.axvspan(0.14, 0.22, color="#B64342", alpha=0.08, linewidth=0)
            ax.axvspan(0.30, 0.60, color="#D9A441", alpha=0.10, linewidth=0)
            ax.set_xlim(-0.05, 0.75)
            if r == 0:
                ax.set_title(roi.replace("_", "\n"), fontsize=6.5)
            if c == 0:
                ax.set_ylabel(f"{COMPONENT_LABELS[component]}\nAmplitude (uV)")
            if r == 1:
                ax.set_xlabel("Time (s)")
            if r == 0 and c == len(rois) - 1:
                ax.legend(frameon=False, fontsize=6, title="Age")
    fig.suptitle("Six-ROI ERP waveforms", y=0.99, fontsize=9, fontweight="bold")
    fig.tight_layout(h_pad=1.0, w_pad=0.7)
    save_pub(fig, out_base)
    plt.close(fig)


def plot_component_roi_heatmap(summary: pd.DataFrame, out_base: Path) -> None:
    rois = list(SIX_ROIS)
    erp_components = list(ERP_WINDOWS)
    fig, axes = plt.subplots(2, len(AGE_GROUPS), figsize=(10.4, 5.8), sharex=True, sharey=True)
    vmax = max(0.1, float(np.nanpercentile(np.abs(summary["mean_uv"]), 98)))
    last_im = None
    for r, component in enumerate(COMPONENTS):
        for c, age_group in enumerate(AGE_GROUPS):
            ax = axes[r, c]
            sub = summary[summary["component"].eq(component) & summary["age_group"].eq(age_group)]
            mat = (
                sub.pivot(index="roi", columns="erp_component", values="mean_uv")
                .reindex(index=rois, columns=erp_components)
                .to_numpy()
            )
            last_im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(f"{COMPONENT_LABELS[component]}\nAge {AGE_LABELS[age_group]}", fontsize=7)
            if c == 0:
                ax.set_yticks(np.arange(len(rois)))
                ax.set_yticklabels([roi.replace("_", " ") for roi in rois], fontsize=6)
            else:
                ax.set_yticks(np.arange(len(rois)))
                ax.set_yticklabels([])
            if r == 1:
                ax.set_xticks(np.arange(len(erp_components)))
                ax.set_xticklabels(erp_components, rotation=45, ha="right", fontsize=6)
            else:
                ax.set_xticks(np.arange(len(erp_components)))
                ax.set_xticklabels([])
    fig.subplots_adjust(left=0.14, right=0.91, bottom=0.13, top=0.86, wspace=0.08, hspace=0.18)
    cax = fig.add_axes([0.93, 0.22, 0.018, 0.52])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("Mean amplitude (uV)", rotation=270, labelpad=10)
    fig.suptitle("Six-ROI ERP window amplitudes", y=0.97, fontsize=9, fontweight="bold")
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
    features, group_waveforms = build_features_and_group_waveforms(components, times, ch_names)
    summary = summarize_features(features)
    roi_defs = roi_definition_table(ch_names)

    roi_defs.to_csv(out_dir / "six_roi_definitions.csv", index=False)
    features.to_csv(out_dir / "six_roi_erp_features_by_subject.csv", index=False)
    summary.to_csv(out_dir / "six_roi_erp_group_summary.csv", index=False)
    group_waveforms.to_csv(out_dir / "six_roi_erp_group_waveforms.csv", index=False)
    recording_summary.to_csv(out_dir / "six_roi_cached_recording_summary.csv", index=False)

    plot_roi_waveforms(group_waveforms, figures_dir / "six_roi_erp_waveforms")
    plot_component_roi_heatmap(summary, figures_dir / "six_roi_erp_window_heatmap")

    print(roi_defs.groupby("roi")["channel"].apply(lambda x: ",".join(x)).to_string())
    print(summary.head(18).to_string(index=False))
    print(f"subjects: {features['subject'].nunique()}")
    print(f"rows: {len(features)}")


if __name__ == "__main__":
    main()

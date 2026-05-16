# Script Index

This open-source package contains the ERP manuscript code only. Resting-state
time-irreversibility scripts and outputs are intentionally excluded.

## `s00`: Shared Utilities

| Script | Purpose |
| --- | --- |
| `s00_utils_common.py` | Shared config, path, and naming helpers imported by multiple scripts. |

## `s01-s07`: Data Setup and QC

| Script | Purpose |
| --- | --- |
| `s01_data_build_manifest.py` | Build a BIDS recording manifest from configured HBN roots. |
| `s02_data_prepare_clean_manifest.py` | Clean and filter raw manifest outputs. |
| `s03_data_classify_age_groups.py` | Assign participants to developmental age groups. |
| `s04_data_inspect_bids_format.py` | Inspect local BIDS/event formats before analysis. |
| `s05_qc_mne_preflight_check.py` | Lightweight MNE/BIDS readiness check. |
| `s06_data_build_event_taxonomy.py` | Generate the event taxonomy table used by the workflow scaffold. |
| `s07_qc_compute_event_statistics.py` | Summarise event counts and task-level event availability. |

## `s10-s14`: ERP Preprocessing and Feature Extraction

| Script | Purpose |
| --- | --- |
| `s10_preprocess_visual_epochs.py` | Preprocess SuS/CCD visual-onset epochs. |
| `s11_extract_epoch_erp_features.py` | Extract subject-task ERP window features. |
| `s12_extract_selected_electrode_features.py` | Extract selected-electrode ERP features for waveform/topography figures. |
| `s13_extract_six_roi_features.py` | Exploratory six-ROI extraction. |
| `s14_qc_split_half_reliability.py` | Split-half reliability helper. |

## `s20-s26`: Statistics and Sensitivity Analyses

| Script | Purpose |
| --- | --- |
| `s20_model_component_decomposition.py` | Component decomposition helpers used by downstream analyses. |
| `s21_stats_full_sus_ccd_followup.py` | Full SuS+CCD developmental follow-up analyses. |
| `s22_stats_compare_general_specific_development.py` | General-vs-specific developmental contrast tests. |
| `s23_stats_psychopathology_age_interactions.py` | Psychopathology-by-age interaction tests. |
| `s24_stats_trial_variability_rescue.py` | Trial-to-trial variability rescue analysis. |
| `s25_stats_ccd_behavior_mediation.py` | CCD behavioural mediation screen. |
| `s26_sensitivity_full_sample_ica.py` | Full-sample ICA sensitivity analysis. |

## `s30-s38`: Figures and Supplement

| Script | Output naming |
| --- | --- |
| `s30_fig1_study_design_flowchart.py` | `fig1.*` |
| `s31_fig2_fig3_fig4_fig6_core_results.py` | `fig2.*`, `fig3.*`, `fig4.*`, `fig6.*` |
| `s32_fig5_erp_waveform_topomap_grid.py` | `fig5.*` |
| `s33_figS1_figS3_companion_panels.py` | `figS1.*` to `figS3.*` |
| `s34_figures_full_channel_topomaps.py` | full-channel ERP checks |
| `s35_figures_component_waveform_topomap.py` | component waveform/topomap panels |
| `s36_figures_extended_component_checks.py` | exploratory component panels |
| `s37_supplement_build_all_materials.py` | `figS1.*` to `figS9.*` and supplementary tables |
| `s38_supplement_render_tables_docx.py` | supplementary table DOCX rendering |

## `s50-s56`: Snakemake Scaffold

These scripts support the lightweight `workflow/Snakefile` scaffold. The
Snakefile also calls the manifest and event-taxonomy scripts listed above. The
current manuscript analyses primarily used the explicit scripts above, but the
scaffold is kept for users who prefer workflow managers.

| Script | Purpose |
| --- | --- |
| `s50_workflow_preprocess_scaffold.py` | early preprocessing placeholder/scaffold |
| `s51_workflow_deconvolution_scaffold.py` | deconvolution scaffold |
| `s52_workflow_extract_features_scaffold.py` | feature extraction scaffold |
| `s53_workflow_variance_decomposition.R` | R variance-decomposition scaffold |
| `s54_workflow_developmental_gamm.R` | R developmental model scaffold |
| `s55_workflow_psychopathology_assoc.R` | R psychopathology association scaffold |
| `s56_workflow_qc_report.py` | QC report scaffold |

## Excluded

The following exploratory scripts are not part of this open-source package:

- `resting_time_irreversibility.py`
- `plot_time_irreversibility_results.py`
- `plot_time_irreversibility_manuscript_figures.py`

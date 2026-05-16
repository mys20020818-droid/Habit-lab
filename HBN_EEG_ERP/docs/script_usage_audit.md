# Script Usage Audit

This audit checks whether files in `scripts/` are part of the open-source ERP
release, are imported by other scripts, or are only optional/scaffold code.

## Removed

| Script | Reason |
| --- | --- |
| `visualize_full_sus_ccd_results.py` | Old visualization script. It was not referenced by README, docs, the Snakemake workflow, or any other script in the open-source package. Its role is superseded by `s31_fig2_fig3_fig4_fig6_core_results.py` and `s33_figS1_figS3_companion_panels.py`. |

## Required for the Main ERP Manuscript Path

These scripts are either direct entry points or dependencies for reproducing
the main SuS+CCD ERP analysis and manuscript figures.

| Script | Role |
| --- | --- |
| `s00_utils_common.py` | Shared helpers for config loading, path creation, and naming. |
| `s01_data_build_manifest.py` | Build BIDS recording manifest. |
| `s03_data_classify_age_groups.py` | Age-bin helper used by data inspection and manifest summaries. |
| `s04_data_inspect_bids_format.py` | BIDS/event inspection before preprocessing. |
| `s10_preprocess_visual_epochs.py` | Current visual-epoch preprocessing entry point. |
| `s11_extract_epoch_erp_features.py` | Current ERP window feature extraction. |
| `s20_model_component_decomposition.py` | Provides component decomposition helpers used by follow-up and ICA scripts. |
| `s21_stats_full_sus_ccd_followup.py` | Main follow-up developmental/statistical summaries. |
| `s22_stats_compare_general_specific_development.py` | Main general-vs-specific developmental contrast analysis. |
| `s23_stats_psychopathology_age_interactions.py` | Psychopathology-by-age interaction analysis. |
| `s24_stats_trial_variability_rescue.py` | Trial-to-trial variability rescue analysis. |
| `s25_stats_ccd_behavior_mediation.py` | CCD behavioural mediation screen. |
| `s26_sensitivity_full_sample_ica.py` | ICA sensitivity workflow. |
| `s30_fig1_study_design_flowchart.py` | Generates `fig1`. |
| `s31_fig2_fig3_fig4_fig6_core_results.py` | Generates `fig2`, `fig3`, `fig4`, and `fig6`. |
| `s12_extract_selected_electrode_features.py` | Supports selected-electrode waveform/topography figure generation. |
| `s34_figures_full_channel_topomaps.py` | Shared full-channel/topomap utilities used by selected-electrode plotting scripts. |
| `s32_fig5_erp_waveform_topomap_grid.py` | Generates `fig5`. |
| `s33_figS1_figS3_companion_panels.py` | Generates companion/supplement-style `figS1` to `figS3`. |
| `s37_supplement_build_all_materials.py` | Generates full supplementary tables and `figS1` to `figS9`. |
| `s38_supplement_render_tables_docx.py` | Renders supplementary tables to DOCX when needed. |

## Optional but Useful

These are not required for the shortest manuscript rerun, but are useful for
QC, exploratory checks, or reviewer follow-up. Keep them unless you want a very
minimal repository.

| Script | Reason to keep |
| --- | --- |
| `s02_data_prepare_clean_manifest.py` | Useful for cleaning manifest variants during data setup. |
| `s05_qc_mne_preflight_check.py` | Quick environment/BIDS readiness check. |
| `s07_qc_compute_event_statistics.py` | Event-count summary helper. |
| `s14_qc_split_half_reliability.py` | Small reliability helper script. |
| `s13_extract_six_roi_features.py` | Optional ROI-level sensitivity/exploratory extraction. |
| `s35_figures_component_waveform_topomap.py` | Optional waveform/topomap panel style. |
| `s36_figures_extended_component_checks.py` | Optional extended component checks. |

## Snakemake Scaffold

These files are used by `workflow/Snakefile`. They are not the current explicit
manuscript rerun path, but they keep the workflow scaffold self-contained.

| Script | Workflow role |
| --- | --- |
| `s50_workflow_preprocess_scaffold.py` | Snakemake preprocessing scaffold. |
| `s06_data_build_event_taxonomy.py` | Snakemake event taxonomy step. |
| `s51_workflow_deconvolution_scaffold.py` | Snakemake deconvolution scaffold. |
| `s52_workflow_extract_features_scaffold.py` | Snakemake feature extraction scaffold. |
| `s53_workflow_variance_decomposition.R` | Snakemake variance-decomposition step. |
| `s54_workflow_developmental_gamm.R` | Snakemake developmental model step. |
| `s55_workflow_psychopathology_assoc.R` | Snakemake psychopathology association step. |
| `s56_workflow_qc_report.py` | Snakemake QC report step. |

If you want the leanest possible GitHub repository, remove `workflow/` and the
Snakemake scaffold scripts above. If you want a workflow-manager example, keep
them.

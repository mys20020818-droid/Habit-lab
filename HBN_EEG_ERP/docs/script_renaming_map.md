# Script Renaming Map

The `scripts/` folder was renamed into numbered, stage-based entry points so
that the workflow sorts naturally in file browsers.

| Old script | New script |
| --- | --- |
| `common.py` | `s00_utils_common.py` |
| `build_manifest.py` | `s01_data_build_manifest.py` |
| `prepare_clean_manifest.py` | `s02_data_prepare_clean_manifest.py` |
| `classify_age_groups.py` | `s03_data_classify_age_groups.py` |
| `inspect_data_format.py` | `s04_data_inspect_bids_format.py` |
| `mne_preflight_check.py` | `s05_qc_mne_preflight_check.py` |
| `build_event_taxonomy.py` | `s06_data_build_event_taxonomy.py` |
| `compute_event_stats.py` | `s07_qc_compute_event_statistics.py` |
| `preprocess_visual_epochs.py` | `s10_preprocess_visual_epochs.py` |
| `extract_epoch_erp_features.py` | `s11_extract_epoch_erp_features.py` |
| `extract_selected_electrode_erp_features.py` | `s12_extract_selected_electrode_features.py` |
| `extract_six_roi_erp_features.py` | `s13_extract_six_roi_features.py` |
| `compute_split_half_reliability.py` | `s14_qc_split_half_reliability.py` |
| `pilot_decomposition_analysis.py` | `s20_model_component_decomposition.py` |
| `full_sus_ccd_followup_analysis.py` | `s21_stats_full_sus_ccd_followup.py` |
| `compare_general_specific_development.py` | `s22_stats_compare_general_specific_development.py` |
| `pfactor_age_interaction_analysis.py` | `s23_stats_psychopathology_age_interactions.py` |
| `clinical_rescue_iiv_analysis.py` | `s24_stats_trial_variability_rescue.py` |
| `clinical_rescue_ccd_behavior_mediation.py` | `s25_stats_ccd_behavior_mediation.py` |
| `ica_sensitivity_analysis.py` | `s26_sensitivity_full_sample_ica.py` |
| `plot_manuscript_figure1_flowchart.py` | `s30_fig1_study_design_flowchart.py` |
| `plot_manuscript_core_figures.py` | `s31_fig2_fig3_fig4_fig6_core_results.py` |
| `plot_main_erp_waveform_topomap_grid.py` | `s32_fig5_erp_waveform_topomap_grid.py` |
| `plot_nature_companion_figures.py` | `s33_figS1_figS3_companion_panels.py` |
| `plot_full_channel_erp_topomaps.py` | `s34_figures_full_channel_topomaps.py` |
| `plot_component_reference_style_waveform_topomap.py` | `s35_figures_component_waveform_topomap.py` |
| `plot_extended_full_channel_erp_components.py` | `s36_figures_extended_component_checks.py` |
| `complete_supplementary_materials.py` | `s37_supplement_build_all_materials.py` |
| `render_supplementary_tables_docx.py` | `s38_supplement_render_tables_docx.py` |
| `preprocess_mne.py` | `s50_workflow_preprocess_scaffold.py` |
| `deconvolve.py` | `s51_workflow_deconvolution_scaffold.py` |
| `extract_features.py` | `s52_workflow_extract_features_scaffold.py` |
| `variance_decomposition.R` | `s53_workflow_variance_decomposition.R` |
| `developmental_gamm.R` | `s54_workflow_developmental_gamm.R` |
| `psychopathology_assoc.R` | `s55_workflow_psychopathology_assoc.R` |
| `qc_report.py` | `s56_workflow_qc_report.py` |

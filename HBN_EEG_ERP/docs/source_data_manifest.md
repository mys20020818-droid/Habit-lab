# Source Data Manifest

This file links manuscript-facing figures to the scripts and CSV source data
included in this open-source package.

## Main Figures

| Figure | Output base | Build script | Source data |
| --- | --- | --- | --- |
| Figure 1 | `figures/main/fig1` | `scripts/s30_fig1_study_design_flowchart.py` | `source_data/manuscript_core_nature/figure1_flowchart_nodes.csv` |
| Figure 2 | `figures/main/fig2` | `scripts/s31_fig2_fig3_fig4_fig6_core_results.py` | `source_data/manuscript_core_nature/figure2_variance_decomposition.csv`; `figure2_component_reliability.csv`; `figure2_task_reliability_reference.csv` |
| Figure 3 | `figures/main/fig3` | `scripts/s31_fig2_fig3_fig4_fig6_core_results.py` | `source_data/manuscript_core_nature/figures3_4_component_scores.csv`; `figures3_4_age_models.csv`; `figure3_age_group_posthoc.csv` |
| Figure 4 | `figures/main/fig4` | `scripts/s31_fig2_fig3_fig4_fig6_core_results.py` | `source_data/manuscript_core_nature/figures3_4_component_scores.csv`; `figures3_4_age_models.csv`; `figures3_4_development_contrast_summary.csv` |
| Figure 5 | `figures/main/fig5` | `scripts/s32_fig5_erp_waveform_topomap_grid.py` | waveform/topography cache files supplied at runtime |
| Figure 6 | `figures/main/fig6` | `scripts/s31_fig2_fig3_fig4_fig6_core_results.py` | `source_data/manuscript_core_nature/figure5_psychopathology_effects.csv`; `figure5_pfactor_tertile_no_interaction_predictions.csv` |

## Supplementary Figures

| Figure | Output base | Build script |
| --- | --- | --- |
| Supplementary Figure S1 | `figures/supplementary/figS1` | `scripts/s37_supplement_build_all_materials.py` |
| Supplementary Figure S2 | `figures/supplementary/figS2` | `scripts/s37_supplement_build_all_materials.py` |
| Supplementary Figure S3 | `figures/supplementary/figS3` | `scripts/s37_supplement_build_all_materials.py` |
| Supplementary Figure S4 | `figures/supplementary/figS4` | `scripts/s37_supplement_build_all_materials.py` |
| Supplementary Figure S5 | `figures/supplementary/figS5` | `scripts/s37_supplement_build_all_materials.py` |
| Supplementary Figure S6 | `figures/supplementary/figS6` | `scripts/s37_supplement_build_all_materials.py` |
| Supplementary Figure S7 | `figures/supplementary/figS7` | `scripts/s37_supplement_build_all_materials.py` |
| Supplementary Figure S8 | `figures/supplementary/figS8` | `scripts/s37_supplement_build_all_materials.py` |
| Supplementary Figure S9 | `figures/supplementary/figS9` | `scripts/s37_supplement_build_all_materials.py` |

## Naming Note

Some source CSV filenames retain earlier `figure*` names because the analyses
were generated before final figure numbering was settled. Figure exports use
the clean `fig*` and `figS*` naming convention.

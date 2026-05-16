# HBN SuS+CCD Developmental ERP Decomposition

This repository contains the analysis and figure-generation code for a
two-task developmental ERP decomposition of Healthy Brain Network EEG data.
The analysis focuses on visual-onset ERP responses shared between the passive
surround-suppression task (SuS) and the active contrast-change-detection task
(CCD).

The repository is intentionally ERP-only. Exploratory resting-state
time-irreversibility analyses are not included.

## Main Components

The manuscript-facing ERP labels are:

| Label | Time window | Channel group |
| --- | --- | --- |
| P1 | 80-140 ms | `visual_posterior` |
| P2 | 140-220 ms | `visual_posterior` |
| N450 | 300-600 ms | `frontocentral` |

Internal feature keys are retained for compatibility with existing source-data
tables. Use the display-label mapping in `config/config.example.yaml` when
rendering manuscript outputs.

## Repository Layout

```text
config/
  config.example.yaml          # template without local paths
docs/
  script_index.md              # what each script does
  source_data_manifest.md      # figure-to-source-data map
  reproducibility_checklist.md # pre-publication checklist
figures/
  main/                        # fig1-fig6, no TIFF files
  supplementary/               # figS1-figS9, no TIFF files
scripts/
  s*.py, s*.R                  # numbered analysis, QC, plotting, and supplement scripts
source_data/
  manuscript_core_nature/      # CSV source data for main figures
  nature_companion/            # CSV source data for companion/supplement panels
  ica_sensitivity/             # CSV outputs for ICA sensitivity checks
workflow/
  Snakefile
  envs/hbn-eeg.yml
```

## Data

Raw HBN EEG data are not included. Download the relevant HBN EEG BIDS releases
from their public repository and update `config/config.example.yaml` with local
paths before running the preprocessing scripts.

Large derived epochs, local derivatives, DOCX drafts, and machine-specific
intermediate files are not included. Lightweight CSV source data used for the
figures are included under `source_data/`.

## Environment

Create the conda environment:

```bash
conda env create -f workflow/envs/hbn-eeg.yml
conda activate hbn-eeg
```

Alternatively, install the key Python packages manually:

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn mne mne-bids pyyaml python-docx openpyxl
```

## Typical Workflow

1. Copy `config/config.example.yaml` to `config/config.yaml`.
2. Edit `config/config.yaml` so that `paths.bids_roots` points to your local HBN
   BIDS release folders.
3. Build or inspect the local manifest:

```bash
python scripts/s01_data_build_manifest.py --config config/config.yaml --out results/tables/manifest.csv
python scripts/s04_data_inspect_bids_format.py --config config/config.yaml --manifest results/tables/manifest.csv
```

4. Run preprocessing and feature extraction as needed:

```bash
python scripts/s10_preprocess_visual_epochs.py --config config/config.yaml --manifest results/tables/manifest.csv --qc-out results/tables/visual_epoch_qc.csv
python scripts/s11_extract_epoch_erp_features.py --config config/config.yaml --qc results/tables/visual_epoch_qc.csv --recording-out results/tables/recording_features.csv --subject-task-out results/tables/subject_task_features.csv
```

5. Rebuild figures from available source data:

```bash
python scripts/s30_fig1_study_design_flowchart.py
python scripts/s31_fig2_fig3_fig4_fig6_core_results.py
```

Figure 5 requires waveform/topography cache files supplied with
`--waveforms`, `--window-values`, and `--cache-dir`; see
`docs/source_data_manifest.md`.

## Figure Naming

Main manuscript figures are exported as `fig1` to `fig6`.
Supplementary figures are exported as `figS1` to `figS9`.

## Not Included

The following are intentionally excluded from this open-source package:

- raw HBN BIDS data,
- large derived EEG epoch files,
- TIFF exports,
- Word manuscript drafts,
- local machine paths,
- exploratory resting-state time-irreversibility code and outputs.

## Citation

If you use this code, please cite the associated manuscript and the Healthy
Brain Network data resource. Fill in `CITATION.cff` after the manuscript DOI is
available.

## License

No open-source license is applied until the author chooses one. A MIT license
template is provided in `LICENSE_TEMPLATE_MIT.txt`.

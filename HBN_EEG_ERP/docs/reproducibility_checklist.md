# Reproducibility Checklist

## Before Running

- [ ] Download the relevant HBN EEG BIDS releases.
- [ ] Copy `config/config.example.yaml` to `config/config.yaml`.
- [ ] Replace placeholder BIDS roots with local paths.
- [ ] Create the `hbn-eeg` conda environment.
- [ ] Confirm the preprocessing defaults: 1-40 Hz band-pass, 60 Hz notch,
      average reference, 250 microvolt epoch rejection.

## ERP Analyses

- [ ] Build or verify the SuS+CCD manifest.
- [ ] Check event labels for SuS and CCD visual-onset events.
- [ ] Run visual-epoch preprocessing.
- [ ] Extract P1/P2/N450 component amplitudes.
- [ ] Recompute task-general and task-specific component scores.
- [ ] Run developmental models and general-vs-specific contrasts.
- [ ] Run psychopathology association and interaction checks if phenotype data
      are available.
- [ ] Run ICA sensitivity checks if reporting ICA robustness.

## Figures

- [ ] Main figures export as `fig1` to `fig6`.
- [ ] Supplementary figures export as `figS1` to `figS9`.
- [ ] Source CSVs are documented in `docs/source_data_manifest.md`.
- [ ] TIFF files are regenerated locally if required by a journal; they are not
      stored in this lightweight package.

## Open-Source Hygiene

- [ ] No raw HBN data are committed.
- [ ] No subject-level sensitive phenotype files beyond approved source data are
      committed.
- [ ] No local absolute paths are committed.
- [ ] No DOCX manuscript drafts are committed.
- [ ] No resting-state time-irreversibility scripts or outputs are committed.
- [ ] A final license is selected before public release.

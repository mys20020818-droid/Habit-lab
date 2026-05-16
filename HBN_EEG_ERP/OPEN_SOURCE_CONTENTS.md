# Open-Source Package Contents

This package is prepared for a public ERP-analysis code release.

## Included

- ERP preprocessing and feature extraction scripts.
- SuS+CCD developmental and psychopathology analysis scripts.
- ICA sensitivity analysis script.
- Main and supplementary figure scripts.
- Lightweight figure source-data CSV files.
- Existing main and supplementary figure exports in PNG/SVG/PDF format.
- Conda environment file and Snakemake scaffold.
- Configuration template without local machine paths.
- Script usage audit in `docs/script_usage_audit.md`.

## Excluded

- Resting-state time-irreversibility code.
- Resting-state time-irreversibility results.
- Raw HBN EEG data.
- Large derived EEG epoch files.
- TIFF figure exports.
- DOCX manuscript drafts.
- Local absolute data paths.
- Local manuscript-editing utilities.

## Before Publishing

1. Choose and finalize a license.
2. Replace placeholder author and DOI fields in `CITATION.cff`.
3. Review whether included source-data CSVs are approved for public release.
4. Add the final manuscript citation once available.

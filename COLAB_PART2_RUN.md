# PART2 Colab GPU run guide

## Status

`PART2_Colab_Run.ipynb` is kept as a legacy Colab helper. It is not the canonical source for the report numbers.

Use the GCP rerun script and protocol for report reproduction:

```bash
bash scripts/rerun_part2_report_20260611.sh
```

Canonical documentation:

- `docs/part2/05_report_rerun_protocol_20260611.md`
- `results/report_summary_20260611.csv`
- `results/rerun-20260611/paraphrase_experiments.csv`

## Why Colab Is Not Canonical

The original Colab notebook was designed for a longer automatic run and used older defaults such as larger screening subsets and `prompt-full.pt` naming. The final report was verified on the GCP T4 VM with these settings:

- prompt screening: train 5,000 / dev 1,000, 1 epoch
- full training: full train set, direct prompt, 3 epochs
- training batch size 4 with gradient accumulation 2
- max length 128
- bidirectional inference and threshold calibration on dev only
- final test prediction after checkpoint, prompt, and threshold were fixed

## If Colab Must Be Used

Treat Colab output as a separate exploratory rerun unless the notebook is manually updated to exactly match `scripts/rerun_part2_report_20260611.sh`.

Required final artifacts should still have the same meaning:

```text
results/paraphrase_experiments.csv
results/error_analysis_para.csv
predictions/para-test-final.csv
```

Do not use Colab-generated numbers in the report unless the command sequence, hyperparameters, checkpoint name, and split usage match the canonical rerun protocol.

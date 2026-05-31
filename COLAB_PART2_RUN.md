# PART2 Colab GPU run guide

Use the notebook `PART2_Colab_Run.ipynb` on Google Colab with GPU enabled.

This version avoids the previous Drive Git-folder problem:

- source code is cloned fresh into `/content/NLP_Study_PART2_CODE` every runtime
- long-lived outputs are stored in `/content/drive/MyDrive/NLP_Study_PART2_OUTPUT`
- prompt screening selects the best prompt automatically
- threshold calibration writes the selected threshold automatically
- no Git commands are run inside the Drive output folder
- any previous `results/paraphrase_experiments.csv` is backed up before a new run starts

## How to run

1. Open:
   <https://colab.research.google.com/github/Y0onSe0/NLP_Study/blob/part2/PART2_Colab_Run.ipynb>
2. Select `Runtime > Change runtime type > GPU`.
3. Run cells from top to bottom.

You do not need to edit the notebook for the normal run.

## Output locations

All important files are saved under:

```text
/content/drive/MyDrive/NLP_Study_PART2_OUTPUT
```

Expected subfolders:

```text
checkpoints/
logs/
predictions/
results/
```

Main final files:

```text
checkpoints/prompt-full.pt
predictions/para-test-final.csv
results/paraphrase_experiments.csv
results/error_analysis_para.csv
```

## Core setup cell

The notebook uses this structure:

```bash
CODE=/content/NLP_Study_PART2_CODE
OUT=/content/drive/MyDrive/NLP_Study_PART2_OUTPUT

rm -rf "$CODE"
git clone -b part2 --single-branch https://github.com/Y0onSe0/NLP_Study.git "$CODE"
mkdir -p "$OUT/checkpoints" "$OUT/predictions" "$OUT/results" "$OUT/logs"
if [ -f "$OUT/results/paraphrase_experiments.csv" ]; then
  cp "$OUT/results/paraphrase_experiments.csv" "$OUT/results/paraphrase_experiments.backup_$(date +%Y%m%d_%H%M%S).csv"
  rm "$OUT/results/paraphrase_experiments.csv"
fi
```

This is intentional. The code clone is disposable, while outputs survive Colab disconnects.

## Package install

The notebook installs the Colab-friendly packages directly:

```bash
pip install -q \
  transformers==4.46.3 \
  tokenizers==0.20.0 \
  einops==0.8.0 \
  sacrebleu==2.5.1 \
  tqdm==4.58.0 \
  scikit-learn \
  pandas
```

## Full automated flow

The notebook runs:

1. GPU and Drive setup
2. fresh `part2` clone into `/content`
3. package install
4. smoke test
5. prompt screening for `baseline`, `direct`, `meaning`
6. automatic best prompt selection by dev accuracy
7. full training with the selected prompt
8. bidirectional dev prediction
9. automatic threshold calibration
10. error analysis
11. final test prediction

Colab can still disconnect because runtime limits are controlled by Google, but checkpoints/logs/results/predictions are written to Drive.

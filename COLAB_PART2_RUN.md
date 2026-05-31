# PART2 Colab GPU run guide

This guide is for running the `part2` branch on Google Colab GPU while keeping useful outputs in Google Drive.

## Local PC power setting already applied

While plugged in, this Windows PC has been set to:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

That means AC-powered sleep, display timeout, and hibernate timeout are disabled. Keep the laptop plugged in.

To restore later:

```powershell
powercfg /change standby-timeout-ac 30
powercfg /change monitor-timeout-ac 10
powercfg /change hibernate-timeout-ac 0
```

## Colab setup

1. Open Google Colab.
2. Runtime > Change runtime type > GPU.
3. Upload/open `PART2_Colab_Run.ipynb`.
4. Run the cells from top to bottom.

Colab has runtime limits, so do not rely on browser keep-alive tricks. The notebook stores checkpoints, predictions, results, and logs in:

```text
/content/drive/MyDrive/NLP_Study_PART2
```

## Core commands used in the notebook

Clone/update:

```bash
BASE=/content/drive/MyDrive/NLP_Study_PART2
git clone -b part2 --single-branch https://github.com/Y0onSe0/NLP_Study.git "$BASE"
cd "$BASE"
git pull --ff-only origin part2
mkdir -p checkpoints predictions results logs
```

Install packages:

```bash
pip install transformers==4.46.3 tokenizers==0.20.0 einops==0.8.0 sacrebleu==2.5.1 explainaboard_client==0.0.7 tqdm==4.58.0
```

Smoke test:

```bash
python paraphrase_detection.py \
  --use_gpu \
  --mode train_dev \
  --epochs 1 \
  --batch_size 2 \
  --max_train_examples 128 \
  --max_dev_examples 128 \
  --prompt_template baseline \
  --output_tag smoke-baseline
```

Prompt screening:

```bash
python paraphrase_detection.py --use_gpu --mode train_dev --epochs 1 --batch_size 8 --max_train_examples 20000 --max_dev_examples 5000 --prompt_template baseline --output_tag screen-baseline
python paraphrase_detection.py --use_gpu --mode train_dev --epochs 1 --batch_size 8 --max_train_examples 20000 --max_dev_examples 5000 --prompt_template direct --output_tag screen-direct
python paraphrase_detection.py --use_gpu --mode train_dev --epochs 1 --batch_size 8 --max_train_examples 20000 --max_dev_examples 5000 --prompt_template meaning --output_tag screen-meaning
```

Full training, after choosing `BEST_PROMPT`:

```bash
python paraphrase_detection.py \
  --use_gpu \
  --mode train_dev \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-5 \
  --prompt_template "$BEST_PROMPT" \
  --output_tag prompt-full \
  --filepath checkpoints/prompt-full.pt
```

After full training, run dev prediction, threshold calibration, error analysis, and final test prediction from the notebook.

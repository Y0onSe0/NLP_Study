#!/usr/bin/env bash
set -euo pipefail

TAG="rerun-20260611"
PYTHON="${PYTHON:-python}"
EXPERIMENT_LOG="results/${TAG}/paraphrase_experiments.csv"
FULL_CKPT="checkpoints/${TAG}/full-direct-paraphrase.pt"

mkdir -p "logs/${TAG}" "results/${TAG}" "predictions/${TAG}" "checkpoints/${TAG}"

run_step() {
  local name="$1"
  shift
  echo "===== ${name} started $(date -Is) ====="
  "$@" 2>&1 | tee "logs/${TAG}/${name}.log"
  echo "===== ${name} finished $(date -Is) ====="
}

run_step screen-baseline "${PYTHON}" paraphrase_detection.py --use_gpu --mode train_dev \
  --epochs 1 --batch_size 4 --grad_accum_steps 2 --lr 1e-5 \
  --max_train_examples 5000 --max_dev_examples 1000 --max_length 128 \
  --prompt_template baseline \
  --output_tag "${TAG}-screen-baseline" \
  --filepath "checkpoints/${TAG}/screen-baseline-paraphrase.pt" \
  --para_dev_out "predictions/${TAG}/para-dev-screen-baseline.csv" \
  --experiment_log "${EXPERIMENT_LOG}"

run_step screen-direct "${PYTHON}" paraphrase_detection.py --use_gpu --mode train_dev \
  --epochs 1 --batch_size 4 --grad_accum_steps 2 --lr 1e-5 \
  --max_train_examples 5000 --max_dev_examples 1000 --max_length 128 \
  --prompt_template direct \
  --output_tag "${TAG}-screen-direct" \
  --filepath "checkpoints/${TAG}/screen-direct-paraphrase.pt" \
  --para_dev_out "predictions/${TAG}/para-dev-screen-direct.csv" \
  --experiment_log "${EXPERIMENT_LOG}"

run_step screen-meaning "${PYTHON}" paraphrase_detection.py --use_gpu --mode train_dev \
  --epochs 1 --batch_size 4 --grad_accum_steps 2 --lr 1e-5 \
  --max_train_examples 5000 --max_dev_examples 1000 --max_length 128 \
  --prompt_template meaning \
  --output_tag "${TAG}-screen-meaning" \
  --filepath "checkpoints/${TAG}/screen-meaning-paraphrase.pt" \
  --para_dev_out "predictions/${TAG}/para-dev-screen-meaning.csv" \
  --experiment_log "${EXPERIMENT_LOG}"

run_step full-direct-train "${PYTHON}" paraphrase_detection.py --use_gpu --mode train_dev \
  --epochs 3 --batch_size 4 --grad_accum_steps 2 --lr 1e-5 \
  --max_length 128 --prompt_template direct \
  --output_tag "${TAG}-full-direct" \
  --filepath "${FULL_CKPT}" \
  --para_dev_out "predictions/${TAG}/para-dev-full-direct.csv" \
  --experiment_log "${EXPERIMENT_LOG}"

run_step full-direct-dev "${PYTHON}" paraphrase_detection.py --use_gpu --mode dev_predict \
  --batch_size 8 --max_length 128 --prompt_template direct \
  --filepath "${FULL_CKPT}" \
  --output_tag "${TAG}-full-direct" \
  --para_dev_out "predictions/${TAG}/para-dev-full-direct.csv" \
  --experiment_log "${EXPERIMENT_LOG}"

run_step full-direct-bi-dev "${PYTHON}" paraphrase_detection.py --use_gpu --mode dev_predict \
  --batch_size 8 --max_length 128 --prompt_template direct --bidirectional --threshold 0.5 \
  --filepath "${FULL_CKPT}" \
  --output_tag "${TAG}-full-direct-bi" \
  --para_dev_out "predictions/${TAG}/para-dev-full-direct-bi.csv" \
  --experiment_log "${EXPERIMENT_LOG}"

run_step full-direct-bi-calib "${PYTHON}" paraphrase_detection.py --use_gpu --mode calibrate_dev \
  --batch_size 8 --max_length 128 --prompt_template direct --bidirectional \
  --threshold_min 0.30 --threshold_max 0.70 --threshold_step 0.01 \
  --filepath "${FULL_CKPT}" \
  --output_tag "${TAG}-full-direct-bi-calib" \
  --para_dev_out "predictions/${TAG}/para-dev-full-direct-bi-calib.csv" \
  --experiment_log "${EXPERIMENT_LOG}"

run_step full-direct-bi-error "${PYTHON}" paraphrase_detection.py --use_gpu --mode error_analysis \
  --batch_size 8 --max_length 128 --prompt_template direct --bidirectional --threshold 0.56 \
  --filepath "${FULL_CKPT}" \
  --output_tag "${TAG}-full-direct-bi-error" \
  --error_analysis_out "results/${TAG}/error_analysis_para.csv" \
  --experiment_log "${EXPERIMENT_LOG}"

run_step final-test "${PYTHON}" paraphrase_detection.py --use_gpu --mode test_predict \
  --batch_size 8 --max_length 128 --prompt_template direct --bidirectional --threshold 0.56 \
  --filepath "${FULL_CKPT}" \
  --output_tag "${TAG}-final" \
  --para_test_out "predictions/${TAG}/para-test-final.csv" \
  --experiment_log "${EXPERIMENT_LOG}"

{
  echo "===== verification $(date -Is) ====="
  tail -n 20 "${EXPERIMENT_LOG}"
  wc -l "predictions/${TAG}/para-test-final.csv"
  wc -l "results/${TAG}/error_analysis_para.csv"
} | tee "logs/${TAG}/verification.log"

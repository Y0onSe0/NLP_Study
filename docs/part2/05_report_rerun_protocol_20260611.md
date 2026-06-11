# PART-II Report Rerun Protocol (2026-06-11)

이 문서는 보고서에 필요한 PART-II train/dev/test 실행을 다시 수행하기 위한 절차를 정리한다.
기존 최종 산출물을 덮지 않기 위해 모든 재실행 산출물은 `rerun-20260611` 태그를 사용한다.

## 실행 원칙

| 단계 | mode | 읽는 split | 목적 |
| --- | --- | --- | --- |
| prompt screening | `train_dev` | train, dev | baseline/direct/meaning prompt 비교 |
| full training + dev single-direction | `train_dev` | train, dev | direct prompt로 최종 checkpoint 재학습 및 threshold 0.5 단방향 dev 성능 확인 |
| dev bidirectional | `dev_predict` | dev | threshold 0.5 양방향 dev 성능 확인 |
| threshold calibration | `calibrate_dev` | dev | dev 기준 threshold 선택 |
| error analysis | `error_analysis` | dev | dev 오류 사례 추출 |
| final test prediction | `test_predict` | test | 최종 test prediction 생성 |

`test_predict` 전에는 test set을 읽지 않는다. Prompt 선택, checkpoint 선택, bidirectional inference, threshold calibration, error analysis는 모두 train/dev 기준으로만 수행한다.

`train_dev` 모드는 학습이 끝난 뒤 같은 설정으로 `predict_dev`를 자동 호출한다. 따라서 full training 직후 단방향 dev prediction row가 이미 기록되며, 같은 checkpoint로 별도 `dev_predict`를 다시 실행하지 않는다.

## 공통 실행 환경

- VM: `gpu-t4-1`
- Zone: `asia-northeast3-c`
- GPU: NVIDIA Tesla T4
- Conda env: `nlp_final`
- Remote project path: `/home/tansanguy1/NLP_Study`
- Experiment log: `results/rerun-20260611/paraphrase_experiments.csv`
- Report summary: `results/report_summary_20260611.csv`

## 재실행 명령

```bash
mkdir -p logs results/rerun-20260611 predictions/rerun-20260611 checkpoints/rerun-20260611

python paraphrase_detection.py --use_gpu --mode train_dev \
  --epochs 1 --batch_size 4 --grad_accum_steps 2 --lr 1e-5 \
  --max_train_examples 5000 --max_dev_examples 1000 --max_length 128 \
  --prompt_template baseline \
  --output_tag rerun-20260611-screen-baseline \
  --filepath checkpoints/rerun-20260611/screen-baseline-paraphrase.pt \
  --para_dev_out predictions/rerun-20260611/para-dev-screen-baseline.csv \
  --experiment_log results/rerun-20260611/paraphrase_experiments.csv

python paraphrase_detection.py --use_gpu --mode train_dev \
  --epochs 1 --batch_size 4 --grad_accum_steps 2 --lr 1e-5 \
  --max_train_examples 5000 --max_dev_examples 1000 --max_length 128 \
  --prompt_template direct \
  --output_tag rerun-20260611-screen-direct \
  --filepath checkpoints/rerun-20260611/screen-direct-paraphrase.pt \
  --para_dev_out predictions/rerun-20260611/para-dev-screen-direct.csv \
  --experiment_log results/rerun-20260611/paraphrase_experiments.csv

python paraphrase_detection.py --use_gpu --mode train_dev \
  --epochs 1 --batch_size 4 --grad_accum_steps 2 --lr 1e-5 \
  --max_train_examples 5000 --max_dev_examples 1000 --max_length 128 \
  --prompt_template meaning \
  --output_tag rerun-20260611-screen-meaning \
  --filepath checkpoints/rerun-20260611/screen-meaning-paraphrase.pt \
  --para_dev_out predictions/rerun-20260611/para-dev-screen-meaning.csv \
  --experiment_log results/rerun-20260611/paraphrase_experiments.csv

python paraphrase_detection.py --use_gpu --mode train_dev \
  --epochs 3 --batch_size 4 --grad_accum_steps 2 --lr 1e-5 \
  --max_length 128 --prompt_template direct \
  --output_tag rerun-20260611-full-direct \
  --filepath checkpoints/rerun-20260611/full-direct-paraphrase.pt \
  --para_dev_out predictions/rerun-20260611/para-dev-full-direct.csv \
  --experiment_log results/rerun-20260611/paraphrase_experiments.csv

python paraphrase_detection.py --use_gpu --mode dev_predict \
  --batch_size 8 --max_length 128 --prompt_template direct --bidirectional --threshold 0.5 \
  --filepath checkpoints/rerun-20260611/full-direct-paraphrase.pt \
  --output_tag rerun-20260611-full-direct-bi \
  --para_dev_out predictions/rerun-20260611/para-dev-full-direct-bi.csv \
  --experiment_log results/rerun-20260611/paraphrase_experiments.csv

python paraphrase_detection.py --use_gpu --mode calibrate_dev \
  --batch_size 8 --max_length 128 --prompt_template direct --bidirectional \
  --threshold_min 0.30 --threshold_max 0.70 --threshold_step 0.01 \
  --filepath checkpoints/rerun-20260611/full-direct-paraphrase.pt \
  --output_tag rerun-20260611-full-direct-bi-calib \
  --experiment_log results/rerun-20260611/paraphrase_experiments.csv

python paraphrase_detection.py --use_gpu --mode error_analysis \
  --batch_size 8 --max_length 128 --prompt_template direct --bidirectional --threshold 0.56 \
  --filepath checkpoints/rerun-20260611/full-direct-paraphrase.pt \
  --output_tag rerun-20260611-full-direct-bi-error \
  --error_analysis_out results/rerun-20260611/error_analysis_para.csv \
  --experiment_log results/rerun-20260611/paraphrase_experiments.csv

python paraphrase_detection.py --use_gpu --mode test_predict \
  --batch_size 8 --max_length 128 --prompt_template direct --bidirectional --threshold 0.56 \
  --filepath checkpoints/rerun-20260611/full-direct-paraphrase.pt \
  --output_tag rerun-20260611-final \
  --para_test_out predictions/rerun-20260611/para-test-final.csv \
  --experiment_log results/rerun-20260611/paraphrase_experiments.csv

python scripts/build_report_summary_20260611.py
```

## 완료 후 확인

```bash
tail -n 20 results/rerun-20260611/paraphrase_experiments.csv
cat results/report_summary_20260611.csv
wc -l predictions/rerun-20260611/para-test-final.csv
wc -l results/rerun-20260611/error_analysis_para.csv
```

`results/rerun-20260611/paraphrase_experiments.csv`는 raw 실행 이력이고, `results/report_summary_20260611.csv`는 보고서 표를 위한 파생 요약이다. Raw CSV는 실행 이력 보존을 위해 직접 수정하지 않는다.

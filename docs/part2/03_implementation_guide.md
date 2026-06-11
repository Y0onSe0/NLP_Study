# 구현 및 실행 가이드

## 1. 관련 파일

- `paraphrase_detection.py`
  `ParaphraseGPT`, 학습 루프, checkpoint 로드, 실행 모드 분기, tokenizer verbalizer 검증, error analysis 저장을 담당한다.

- `datasets.py`
  Quora TSV 데이터를 읽고, `baseline`, `direct`, `meaning` prompt template으로 Cloze-style 입력을 만든다. Train/dev/test dataset 모두 같은 prompt builder를 사용한다.

- `evaluation.py`
  Dev/test prediction, `p_yes` 계산, bidirectional inference, threshold calibration helper를 포함한다.

- `models/gpt2.py`
  GPT-2 backbone 구현을 포함한다. `hidden_state_to_token()`은 마지막 hidden state를 vocabulary logits로 변환한다.

- `results/paraphrase_experiments.csv`
  실행 모드, checkpoint, prompt, threshold, dev accuracy, dev F1, prediction path 등 실험 기록을 저장한다.

- `results/error_analysis_para.csv`
  Dev set 오류 분석 결과를 저장한다. False positive, false negative, borderline case가 포함된다.

- `predictions/para-dev-*.csv`
  Dev prediction 파일이다. 개발 중 성능 비교와 ablation 기록에 사용한다.

- `predictions/para-test-final.csv`
  최종 제출용 test prediction 파일이다. 최종 설정 확정 후 한 번만 생성한다.

- `checkpoints/*.pt`
  학습된 모델 checkpoint이다. Baseline과 prompt 개선 모델을 분리해 저장한다.

## 2. 실행 모드

| mode | train 사용 | dev 사용 | test 사용 | 목적 |
|---|---:|---:|---:|---|
| `train_dev` | O | O | X | 학습 및 dev 평가 |
| `dev_predict` | X | O | X | 저장된 checkpoint로 dev prediction 생성 |
| `calibrate_dev` | X | O | X | dev set에서 threshold 선택 |
| `error_analysis` | X | O | X | dev set 오류 분석 |
| `test_predict` | X | X | O | 최종 제출 파일 생성 |

`test_predict` 외의 모드는 test set을 읽지 않는 것을 원칙으로 한다.

Dev 파일을 읽는 모든 코드 경로는 `split='dev'`를 명시한다. 과거 로그에 `Loaded ... train examples from data/quora-dev.csv`처럼 표시된 경우는 dev 파일을 train으로 사용한 것이 아니라 split 이름 기본값 때문에 생긴 로깅 표기 오류다. 자세한 검증은 [04_data_split_audit.md](04_data_split_audit.md)에 정리한다.

## 3. Tokenizer / Verbalizer 검증 명령

```bash
conda activate nlp_final
python -c "from types import SimpleNamespace; from paraphrase_detection import verify_yes_no_tokens; verify_yes_no_tokens(SimpleNamespace())"
```

## 4. Smoke Test

맥북 CPU용:

```bash
python paraphrase_detection.py \
  --mode train_dev \
  --epochs 1 \
  --batch_size 1 \
  --max_train_examples 128 \
  --max_dev_examples 128 \
  --prompt_template baseline \
  --output_tag smoke-mac
```

CUDA GPU용:

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

## 5. Canonical Report Rerun

보고서 숫자를 재현하는 기준 실행은 아래 스크립트다. 개별 명령을 직접 실행할 수도 있지만, 재현성 확인에는 이 스크립트를 기준으로 삼는다.

```bash
bash scripts/rerun_part2_report_20260611.sh
```

이 스크립트는 `rerun-20260611` 태그 아래에 산출물을 분리해서 저장한다. 자세한 단계별 명령은 [05_report_rerun_protocol_20260611.md](05_report_rerun_protocol_20260611.md)에 정리되어 있다.

## 6. Prompt Screening

세 template을 train 5,000개, dev 1,000개, 1 epoch 조건에서 비교한다.

```bash
python paraphrase_detection.py \
  --use_gpu \
  --mode train_dev \
  --epochs 1 \
  --batch_size 4 \
  --grad_accum_steps 2 \
  --lr 1e-5 \
  --max_train_examples 5000 \
  --max_dev_examples 1000 \
  --max_length 128 \
  --prompt_template baseline \
  --output_tag screen-baseline \
  --filepath checkpoints/screen-baseline-paraphrase.pt \
  --para_dev_out predictions/para-dev-screen-baseline.csv

python paraphrase_detection.py \
  --use_gpu \
  --mode train_dev \
  --epochs 1 \
  --batch_size 4 \
  --grad_accum_steps 2 \
  --lr 1e-5 \
  --max_train_examples 5000 \
  --max_dev_examples 1000 \
  --max_length 128 \
  --prompt_template direct \
  --output_tag screen-direct \
  --filepath checkpoints/screen-direct-paraphrase.pt \
  --para_dev_out predictions/para-dev-screen-direct.csv

python paraphrase_detection.py \
  --use_gpu \
  --mode train_dev \
  --epochs 1 \
  --batch_size 4 \
  --grad_accum_steps 2 \
  --lr 1e-5 \
  --max_train_examples 5000 \
  --max_dev_examples 1000 \
  --max_length 128 \
  --prompt_template meaning \
  --output_tag screen-meaning \
  --filepath checkpoints/screen-meaning-paraphrase.pt \
  --para_dev_out predictions/para-dev-screen-meaning.csv
```

## 7. Full Training

Prompt screening에서 가장 높은 성능을 보인 `direct` prompt를 전체 train set으로 3 epoch 학습한다.

```bash
python paraphrase_detection.py \
  --use_gpu \
  --mode train_dev \
  --epochs 3 \
  --batch_size 4 \
  --grad_accum_steps 2 \
  --lr 1e-5 \
  --max_length 128 \
  --prompt_template direct \
  --output_tag full-direct \
  --filepath checkpoints/full-direct-paraphrase.pt \
  --para_dev_out predictions/para-dev-full-direct.csv
```

`train_dev` 모드는 학습 직후 같은 checkpoint로 단방향 dev prediction을 자동 생성한다. 따라서 같은 설정의 `dev_predict`를 별도로 다시 실행하지 않는다.

## 8. Bidirectional Dev Prediction

```bash
python paraphrase_detection.py \
  --use_gpu \
  --mode dev_predict \
  --batch_size 8 \
  --max_length 128 \
  --filepath checkpoints/full-direct-paraphrase.pt \
  --prompt_template direct \
  --bidirectional \
  --threshold 0.5 \
  --output_tag full-direct-bi \
  --para_dev_out predictions/para-dev-full-direct-bi.csv
```

## 9. Threshold Calibration

```bash
python paraphrase_detection.py \
  --use_gpu \
  --mode calibrate_dev \
  --batch_size 8 \
  --max_length 128 \
  --filepath checkpoints/full-direct-paraphrase.pt \
  --prompt_template direct \
  --bidirectional \
  --threshold_min 0.30 \
  --threshold_max 0.70 \
  --threshold_step 0.01 \
  --output_tag full-direct-bi-calib
```

## 10. Error Analysis

`error_analysis` 모드는 dev set만 사용하며, `results/error_analysis_para.csv`를 생성한다.

```bash
python paraphrase_detection.py \
  --use_gpu \
  --mode error_analysis \
  --batch_size 8 \
  --max_length 128 \
  --filepath checkpoints/full-direct-paraphrase.pt \
  --prompt_template direct \
  --bidirectional \
  --threshold 0.56 \
  --output_tag full-direct-bi-error \
  --error_analysis_out results/error_analysis_para.csv
```

## 11. Final Test Prediction

이 명령은 최종 checkpoint, prompt, threshold가 확정된 뒤 한 번만 실행한다. Test 결과를 보고 다시 threshold나 prompt를 바꾸면 안 된다.

```bash
python paraphrase_detection.py \
  --use_gpu \
  --mode test_predict \
  --batch_size 8 \
  --max_length 128 \
  --filepath checkpoints/full-direct-paraphrase.pt \
  --prompt_template direct \
  --bidirectional \
  --threshold 0.56 \
  --para_test_out predictions/para-test-final.csv \
  --output_tag final-test
```

## 12. 제출 전 확인

```bash
wc -l predictions/para-test-final.csv
head -n 5 predictions/para-test-final.csv
tail -n 5 results/paraphrase_experiments.csv
test -f checkpoints/full-direct-paraphrase.pt && echo checkpoint-ok
python prepare_submit.py
```

## 13. 결과 기록 양식

| 실험명 | prompt_template | bidirectional | threshold | dev accuracy | dev F1 | checkpoint | prediction file | 비고 |
|---|---|---:|---:|---:|---:|---|---|---|
| Prompt screening baseline | `baseline` | X | 0.5 | 0.708 | 0.613 | `checkpoints/rerun-20260611/screen-baseline-paraphrase.pt` | `predictions/rerun-20260611/para-dev-screen-baseline.csv` | train 5,000 / dev 1,000 |
| Prompt screening direct | `direct` | X | 0.5 | 0.747 | 0.721 | `checkpoints/rerun-20260611/screen-direct-paraphrase.pt` | `predictions/rerun-20260611/para-dev-screen-direct.csv` | train 5,000 / dev 1,000 |
| Prompt screening meaning | `meaning` | X | 0.5 | 0.676 | 0.525 | `checkpoints/rerun-20260611/screen-meaning-paraphrase.pt` | `predictions/rerun-20260611/para-dev-screen-meaning.csv` | train 5,000 / dev 1,000 |
| Full direct | `direct` | X | 0.5 | 0.887 | 0.880 | `checkpoints/rerun-20260611/full-direct-paraphrase.pt` | `predictions/rerun-20260611/para-dev-full-direct.csv` | 전체 train, 3 epoch |
| Full direct + bidirectional | `direct` | O | 0.5 | 0.890 | 0.884 | `checkpoints/rerun-20260611/full-direct-paraphrase.pt` | `predictions/rerun-20260611/para-dev-full-direct-bi.csv` | 같은 checkpoint 사용 |
| Full direct + bidirectional + calibration | `direct` | O | 0.56 | 0.892 | 0.884 | `checkpoints/rerun-20260611/full-direct-paraphrase.pt` | `results/rerun-20260611/paraphrase_experiments.csv` | dev에서 threshold 선택 |

## 14. 보고서 반영 포인트

- 어떤 변경이 성능에 영향을 주었는지 ablation table로 설명한다.
- Threshold calibration은 test가 아니라 dev에서만 수행한다.
- Error analysis로 모델의 한계를 정성적으로 설명한다.
- 단어 중복은 높지만 의미가 다른 경우, 표현은 다르지만 의미가 같은 경우, 숫자/조건/고유명사 차이로 의미가 달라지는 경우를 대표 예시로 정리한다.

## 보고서용 요약 문단

실험은 prompt screening, full training, bidirectional dev prediction, threshold calibration, error analysis, final test prediction 순서로 진행한다. 각 단계는 `paraphrase_detection.py`의 실행 모드로 분리되어 있으며, `test_predict` 외의 모드는 test set을 사용하지 않는다. Prompt screening은 세 template을 train 5,000개와 dev 1,000개로 비교하고, 가장 좋은 `direct` prompt만 full training에 사용한다. Full training은 전체 train set, 3 epoch, batch size 4, gradient accumulation 2 조건으로 수행한다. Bidirectional inference와 threshold calibration은 같은 checkpoint 위에서 dev set 기준으로 비교한다. 최종 test prediction은 checkpoint, prompt, threshold가 모두 확정된 뒤 한 번만 생성한다.

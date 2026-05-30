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

## 5. Prompt Screening

세 template을 subset + 1 epoch 조건에서 비교한다.

```bash
python paraphrase_detection.py \
  --use_gpu \
  --mode train_dev \
  --epochs 1 \
  --batch_size 8 \
  --max_train_examples 20000 \
  --max_dev_examples 5000 \
  --prompt_template baseline \
  --output_tag screen-baseline

python paraphrase_detection.py \
  --use_gpu \
  --mode train_dev \
  --epochs 1 \
  --batch_size 8 \
  --max_train_examples 20000 \
  --max_dev_examples 5000 \
  --prompt_template direct \
  --output_tag screen-direct

python paraphrase_detection.py \
  --use_gpu \
  --mode train_dev \
  --epochs 1 \
  --batch_size 8 \
  --max_train_examples 20000 \
  --max_dev_examples 5000 \
  --prompt_template meaning \
  --output_tag screen-meaning
```

## 6. Full Training

Baseline full training:

```bash
python paraphrase_detection.py \
  --use_gpu \
  --mode train_dev \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-5 \
  --prompt_template baseline \
  --output_tag baseline-full \
  --filepath checkpoints/baseline-full.pt
```

Best prompt full training:

```bash
python paraphrase_detection.py \
  --use_gpu \
  --mode train_dev \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-5 \
  --prompt_template <BEST_PROMPT> \
  --output_tag prompt-full \
  --filepath checkpoints/prompt-full.pt
```

## 7. Bidirectional Dev Prediction

```bash
python paraphrase_detection.py \
  --use_gpu \
  --mode dev_predict \
  --filepath checkpoints/prompt-full.pt \
  --prompt_template <BEST_PROMPT> \
  --bidirectional \
  --threshold 0.5 \
  --output_tag prompt-bidir-dev
```

## 8. Threshold Calibration

```bash
python paraphrase_detection.py \
  --use_gpu \
  --mode calibrate_dev \
  --filepath checkpoints/prompt-full.pt \
  --prompt_template <BEST_PROMPT> \
  --bidirectional \
  --threshold_min 0.30 \
  --threshold_max 0.70 \
  --threshold_step 0.01 \
  --output_tag prompt-bidir-calib
```

## 9. Error Analysis

`error_analysis` 모드는 dev set만 사용하며, `results/error_analysis_para.csv`를 생성한다.

```bash
python paraphrase_detection.py \
  --use_gpu \
  --mode error_analysis \
  --filepath checkpoints/prompt-full.pt \
  --prompt_template <BEST_PROMPT> \
  --bidirectional \
  --threshold <DEV_SELECTED_THRESHOLD> \
  --output_tag prompt-bidir-error
```

## 10. Final Test Prediction

이 명령은 최종 checkpoint, prompt, threshold가 확정된 뒤 한 번만 실행한다. Test 결과를 보고 다시 threshold나 prompt를 바꾸면 안 된다.

```bash
python paraphrase_detection.py \
  --use_gpu \
  --mode test_predict \
  --filepath checkpoints/prompt-full.pt \
  --prompt_template <BEST_PROMPT> \
  --bidirectional \
  --threshold <DEV_SELECTED_THRESHOLD> \
  --para_test_out predictions/para-test-final.csv \
  --output_tag final-test
```

## 11. 제출 전 확인

```bash
wc -l predictions/para-test-final.csv
head -n 5 predictions/para-test-final.csv
tail -n 5 results/paraphrase_experiments.csv
test -f checkpoints/prompt-full.pt && echo checkpoint-ok
python prepare_submit.py
```

## 12. 결과 기록 양식

| 실험명 | prompt_template | bidirectional | threshold | dev accuracy | dev F1 | checkpoint | prediction file | 비고 |
|---|---|---:|---:|---:|---:|---|---|---|
| Baseline | `baseline` | X | 0.5 |  |  | `checkpoints/baseline-full.pt` | `predictions/para-dev-baseline-full.csv` | 기본 prompt |
| Prompt 개선 | `<BEST_PROMPT>` | X | 0.5 |  |  | `checkpoints/prompt-full.pt` | `predictions/para-dev-prompt-full.csv` | screening 1위 prompt |
| Prompt 개선 + Bidirectional inference | `<BEST_PROMPT>` | O | 0.5 |  |  | `checkpoints/prompt-full.pt` | `predictions/para-dev-prompt-bidir-dev.csv` | 같은 checkpoint 사용 |
| Prompt 개선 + Bidirectional inference + Threshold calibration | `<BEST_PROMPT>` | O | `<DEV_SELECTED_THRESHOLD>` |  |  | `checkpoints/prompt-full.pt` | `predictions/para-dev-*.csv` | dev에서 threshold 선택 |

## 13. 보고서 반영 포인트

- 어떤 변경이 성능에 영향을 주었는지 ablation table로 설명한다.
- Threshold calibration은 test가 아니라 dev에서만 수행한다.
- Error analysis로 모델의 한계를 정성적으로 설명한다.
- 단어 중복은 높지만 의미가 다른 경우, 표현은 다르지만 의미가 같은 경우, 숫자/조건/고유명사 차이로 의미가 달라지는 경우를 대표 예시로 정리한다.

## 보고서용 요약 문단

실험은 baseline 신뢰성 확인에서 시작해 prompt screening, full training, bidirectional dev prediction, threshold calibration, error analysis, final test prediction 순서로 진행한다. 각 단계는 `paraphrase_detection.py`의 실행 모드로 분리되어 있으며, `test_predict` 외의 모드는 test set을 사용하지 않는다. Prompt screening은 세 template을 subset + 1 epoch로 비교하고, 가장 좋은 prompt만 full training에 사용한다. Bidirectional inference와 threshold calibration은 같은 checkpoint 위에서 dev set 기준으로 비교한다. 최종 test prediction은 checkpoint, prompt, threshold가 모두 확정된 뒤 한 번만 생성한다.

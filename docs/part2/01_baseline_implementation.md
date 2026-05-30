# Baseline 구현 설명

## 1. 문제 정의

PART-II Paraphrase Detection은 Quora question pair 데이터셋에서 두 질문 `q1`, `q2`가 paraphrase인지 예측하는 문제이다. Label `1`은 두 질문이 같은 의미임을 나타내고, label `0`은 같은 의미가 아님을 나타낸다.

일반적인 접근에서는 두 문장의 표현을 입력받아 binary classifier head로 label을 예측할 수 있다. 그러나 이 과제에서는 GPT-2의 next-token prediction 구조를 활용하기 위해 문제를 Cloze-style yes/no 생성 문제로 재구성한다. 즉, 모델은 prompt 뒤에 이어질 다음 token이 `yes`인지 `no`인지 예측하며, 이를 paraphrase 여부로 해석한다.

## 2. 입력 형식

`datasets.py`는 `PARAPHRASE_PROMPT_TEMPLATES`와 `build_paraphrase_prompts()`를 통해 두 질문을 자연어 prompt로 변환한다. 현재 baseline template은 다음 형태이다.

```text
Question 1: "{q1}"
Question 2: "{q2}"
Are these questions asking the same thing? Answer yes or no: 
```

모든 prompt template은 마지막에 공백 하나를 포함한다. 이 trailing space 정책 때문에 다음 token verbalizer는 unspaced `yes`와 `no` token을 사용한다.

## 3. 모델 구조

Baseline 모델은 `paraphrase_detection.py`의 `ParaphraseGPT`가 담당한다. `ParaphraseGPT.forward(input_ids, attention_mask)`는 먼저 `GPT2Model` backbone에 tokenized prompt를 입력한다. `models/gpt2.py`의 `GPT2Model.forward()`는 각 token의 hidden state를 계산하고, attention mask를 이용해 마지막 non-padding token의 hidden state를 반환한다.

그 다음 `ParaphraseGPT.forward()`는 `models/gpt2.py`의 `hidden_state_to_token()`을 호출해 마지막 hidden state를 전체 vocabulary에 대한 logits로 변환한다. 전체 vocabulary logits 중 `no` token logit과 `yes` token logit만 선택하며, 최종 2-class logits은 다음 순서를 따른다.

```text
[no_logit, yes_logit]
```

따라서 label `0`은 `no`, label `1`은 `yes`에 대응한다.

## 4. 학습 방식

학습은 `paraphrase_detection.py`의 `train(args)`에서 수행된다. 각 batch에 대해 모델은 `[no_logit, yes_logit]`을 반환하고, gold label과 `F.cross_entropy()`로 loss를 계산한다.

- gold label이 `1`이면 `yes` token의 logit이 커지도록 학습된다.
- gold label이 `0`이면 `no` token의 logit이 커지도록 학습된다.

수식으로 쓰면 각 예시에 대해 다음과 같이 해석할 수 있다.

```text
loss = CrossEntropy([no_logit, yes_logit], gold_label)
```

## 5. 평가 방식

평가는 `evaluation.py`의 `model_eval_paraphrase()`와 `model_test_paraphrase()`가 담당한다. 두 함수는 `[no_logit, yes_logit]`에 softmax를 적용해 `p_yes`를 계산한다.

```text
p_yes = softmax([no_logit, yes_logit])_yes
```

기본 baseline 기준에서는 `p_yes >= 0.5`이면 paraphrase로 예측하고, 그렇지 않으면 non-paraphrase로 예측한다. 주요 지표는 dev accuracy이며, macro-F1도 보조 지표로 기록한다.

## 6. 현재 baseline에서 발견된 신뢰성 이슈와 해결

초기 구현에서 확인된 신뢰성 이슈는 다음과 같다.

- `yes/no` token id가 `8505`, `3919`로 하드코딩되어 있었다.
- train/dev prompt와 test prompt가 서로 달랐다.
- test dataloader가 `shuffle=True`로 설정되어 제출 예측 순서 안정성이 떨어질 수 있었다.
- 기본 실행 흐름이 train 직후 test prediction까지 생성할 수 있어, 개발 중 test set을 반복 사용할 위험이 있었다.
- `paraphrase_detection_head = nn.Linear(args.d, 2)`가 선언되어 있었지만 실제 forward에서는 사용되지 않았다.

현재 구현은 `verify_yes_no_tokens()`에서 `tokenizer.encode("yes")`, `tokenizer.encode("no")`, `tokenizer.encode(" yes")`, `tokenizer.encode(" no")`를 출력하고, unspaced `yes/no` token이 단일 token인지 검증한다. Prompt 생성은 `datasets.py`의 공통 prompt builder로 통일되었고, train/dev/test dataset이 같은 template 정책을 사용한다. 또한 실행 모드는 `train_dev`, `dev_predict`, `calibrate_dev`, `error_analysis`, `test_predict`로 분리되어 있으며, `test_predict` 외에는 test set을 읽지 않는 구조를 목표로 한다.

## 7. 보고서용 요약 문단

본 프로젝트의 baseline은 Quora question pair의 paraphrase 판정을 GPT-2의 next-token prediction 문제로 변환한다. 두 질문은 자연어 prompt로 구성되며, 모델은 prompt 마지막 위치에서 `yes` 또는 `no` token을 예측한다. `ParaphraseGPT`는 GPT-2 backbone의 마지막 non-padding token hidden state를 vocabulary logits로 변환한 뒤, `no`와 `yes` token logit만 선택해 `[no_logit, yes_logit]` 형태의 이진 분류 logits로 사용한다. 학습은 gold label과 cross-entropy loss를 계산하는 방식으로 이루어지며, label `1`은 `yes`, label `0`은 `no`에 대응한다. 평가에서는 softmax로 `p_yes`를 계산하고 기본적으로 threshold `0.5`를 사용한다. 구현 신뢰성을 높이기 위해 verbalizer token id를 tokenizer로 검증하고, prompt builder를 train/dev/test에 공통 적용하며, test set은 최종 `test_predict` 모드에서만 사용하도록 실행 흐름을 분리하였다.

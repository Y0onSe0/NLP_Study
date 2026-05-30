# 성능향상 방법론

## 1. 개선 전략 개요

본 프로젝트의 개선 전략은 모델 구조를 크게 바꾸지 않고, GPT-2의 next-token prediction 특성에 맞게 입력과 판정 방식을 개선하는 것이다. 핵심 개선은 Prompt Template, Verbalizer 검증, Bidirectional Inference, Threshold Calibration, Error Analysis이다. 이러한 방법은 모두 train/dev를 기반으로 선택하고 검증하며, test set은 최종 prediction 생성에만 사용한다.

## 2. Prompt Template 개선

GPT-2는 분류 전용 모델이 아니라 다음 token을 예측하는 언어 모델이다. 따라서 두 질문을 어떤 자연어 prompt로 제시하는지가 `yes/no` 확률에 영향을 줄 수 있다. 현재 구현은 `datasets.py`의 `PARAPHRASE_PROMPT_TEMPLATES`에서 `baseline`, `direct`, `meaning` 세 template만 제공한다. Template 수를 제한하는 이유는 dev set에 대한 prompt overfitting을 줄이고, 보고서에서 해석 가능한 ablation을 유지하기 위해서이다.

Prompt screening은 subset + 1 epoch로 수행하고, 가장 좋은 prompt 하나만 full training에 사용한다.

### baseline

```text
Question 1: "{q1}"
Question 2: "{q2}"
Are these questions asking the same thing? Answer yes or no: 
```

### direct

```text
Is "{q2}" a paraphrase of "{q1}"? Answer yes or no: 
```

### meaning

```text
Do the following two questions have the same meaning?
Question 1: "{q1}"
Question 2: "{q2}"
Answer yes or no: 
```

대표 논문:

- PET: Exploiting Cloze-Questions for Few-Shot Text Classification and Natural Language Inference  
  링크: https://aclanthology.org/2021.eacl-main.20/  
  요약: 분류 문제를 cloze-style 문장으로 바꿔 pretrained language model이 label word를 예측하도록 하는 방법론을 제안한다.
- LM-BFF: Making Pre-trained Language Models Better Few-shot Learners  
  링크: https://aclanthology.org/2021.acl-long.295/  
  요약: prompt 선택과 prompt-based fine-tuning이 pretrained language model의 few-shot 성능에 큰 영향을 줄 수 있음을 보인다.

## 3. Yes/No Verbalizer 검증

모델은 문자열 `"yes"`와 `"no"` 자체가 아니라 tokenizer가 부여한 token id를 예측한다. 따라서 prompt가 trailing space로 끝나는 경우, 다음 token 후보를 unspaced `yes/no`로 둘 것인지 spaced ` yes/ no`로 둘 것인지 확인해야 한다.

현재 구현은 `paraphrase_detection.py`의 `verify_yes_no_tokens()`에서 다음 네 가지 encoding을 모두 출력한다.

- `tokenizer.encode("yes")`
- `tokenizer.encode("no")`
- `tokenizer.encode(" yes")`
- `tokenizer.encode(" no")`

현재 정책은 prompt 끝에 trailing space를 두고, unspaced `yes/no` token을 verbalizer로 사용하는 것이다. 이 정책은 `datasets.py`의 prompt template 주석과 `verify_yes_no_tokens()` 출력에 명시되어 있다.

대표 논문:

- Knowledgeable Prompt-tuning / Incorporating Knowledge into Prompt Verbalizer for Text Classification  
  링크: https://arxiv.org/abs/2108.02035  
  요약: prompt-based classification에서 label word, 즉 verbalizer 선택이 성능과 편향에 영향을 줄 수 있음을 다룬다.

## 4. Bidirectional Inference

Paraphrase 관계는 의미적으로 대칭 관계이다. 즉, `q1`이 `q2`의 paraphrase이면 `q2`도 `q1`의 paraphrase여야 한다. 그러나 GPT-2는 left-to-right decoder-only 모델이므로 입력 순서에 따라 마지막 hidden state와 `yes/no` 확률이 달라질 수 있다.

이를 완화하기 위해 inference 단계에서 `(q1, q2)`와 `(q2, q1)`을 모두 평가하고 `p_yes`를 평균한다. 현재 구현은 `evaluation.py`의 `_batch_yes_probs()`에서 `bidirectional=True`일 때 reverse prompt를 추가로 만들어 평균을 계산한다. 학습 데이터 증강은 비용이 크므로 1차 구현에서는 inference-only로 적용한다.

```text
p_yes_final = (p_yes(q1, q2) + p_yes(q2, q1)) / 2
```

대표 논문:

- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks  
  링크: https://arxiv.org/abs/1908.10084  
  요약: 문장쌍 의미 유사도 판단에서 두 문장 표현의 관계를 비교하는 접근을 제안한다. 본 프로젝트에서는 SBERT를 직접 사용하지 않지만, 문장쌍 의미 관계가 순서에 덜 민감해야 한다는 논리적 근거로 활용할 수 있다.

## 5. Threshold Calibration

기본 판정 기준 `p_yes >= 0.5`가 항상 최적이라고 보장할 수는 없다. 모델이 `yes` 또는 `no` 쪽으로 편향될 수 있으므로, dev set에서 threshold를 탐색한다.

현재 구현은 `evaluation.py`의 `best_threshold_for_accuracy()`에서 threshold 범위 `0.30~0.70`, step `0.01`을 탐색한다. Dev accuracy가 가장 높은 threshold를 선택하고, 동률이면 `0.50`에 가까운 값을 선택한다. 선택된 threshold는 test에는 고정 적용해야 하며, test 결과를 보고 다시 조정하지 않는다.

대표 논문:

- On Calibration of Modern Neural Networks  
  링크: https://arxiv.org/abs/1706.04599  
  요약: 신경망의 confidence가 실제 정답 가능성과 잘 맞지 않을 수 있으며, 후처리 보정이 필요할 수 있음을 보인다.

## 6. Error Analysis

성능 수치만으로는 모델의 실패 양상을 충분히 설명하기 어렵다. 따라서 `paraphrase_detection.py`의 `error_analysis(args)`는 dev set에서 false positive, false negative, threshold 근처 borderline case를 샘플링해 `results/error_analysis_para.csv`로 저장한다.

현재 error type은 다음 heuristic으로 1차 분류한다.

- `number_or_condition`
- `entity_or_name`
- `similar_words_different_meaning`
- `different_expression_same_meaning`
- `needs_manual_review`

이 자동 분류는 보고서 작성의 출발점이며, 보고서에는 대표 사례를 사람이 확인한 뒤 사용하는 것이 적절하다.

## 7. 보고서용 요약 문단

본 프로젝트의 성능향상 방법론은 GPT-2를 큰 폭으로 변경하기보다, Cloze-style paraphrase detection의 입력과 판정 방식을 정교화하는 데 초점을 둔다. 먼저 prompt template을 세 가지로 제한해 비교함으로써, 두 질문을 어떤 자연어 질문 형태로 제시하는지가 `yes/no` 확률에 미치는 영향을 확인한다. 다음으로 verbalizer 검증을 통해 모델이 실제로 비교하는 token id가 의도한 `yes`와 `no`인지 확인한다. Paraphrase 관계의 대칭성을 반영하기 위해 inference 단계에서 `(q1, q2)`와 `(q2, q1)`을 모두 평가하고 `p_yes`를 평균한다. 또한 기본 threshold `0.5`가 최적이 아닐 수 있으므로, dev set에서 threshold를 calibration한다. 마지막으로 false positive, false negative, borderline case를 분석해 모델이 어떤 유형의 질문쌍에서 실패하는지 정성적으로 설명한다. 이 모든 선택은 train/dev에서만 이루어지며, test set은 최종 제출 prediction 생성에만 사용한다.

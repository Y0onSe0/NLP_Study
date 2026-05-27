# PART-I 구현 인수인계 문서

## 1. 구현 완료 파일 목록

| 파일 | 구현 내용 | 테스트 |
|------|-----------|--------|
| `optimizer.py` | AdamW step() 함수 | `optimizer_test.py` ✅ |
| `modules/attention.py` | CausalSelfAttention.attention() | `sanity_check.py` ✅ |
| `modules/gpt2_layer.py` | GPT2Layer.add(), forward() | `sanity_check.py` ✅ |
| `models/gpt2.py` | embed(), hidden_state_to_token() | `sanity_check.py` ✅ |
| `classifier.py` | GPT2SentimentClassifier.__init__(), forward() | 훈련 실행 ✅ |

---

## 2. 각 파일 구현 핵심 내용

### 2.1 `optimizer.py` — AdamW

AdamW 알고리즘 구현 (논문: https://arxiv.org/abs/1412.6980)

```
1. state 초기화: step count(t), 1차 모멘트(m), 2차 모멘트(v)
2. m = β₁·m + (1-β₁)·grad
3. v = β₂·v + (1-β₂)·grad²
4. Bias correction: step_size = lr · √(1-β₂ᵗ) / (1-β₁ᵗ)
5. p = p - step_size · m / (√v + ε)
6. Weight decay: p = p - lr · weight_decay · p  (업데이트 이후 적용)
```

### 2.2 `modules/attention.py` — Causal Self-Attention

**핵심 포인트: Causal Mask 추가**

`utils.py`의 `get_extended_attention_mask`는 padding mask만 처리하고 causal mask를 생성하지 않음.  
GPT-2는 decoder-only 모델이므로 각 토큰이 미래 토큰을 보지 못하도록 `attention()` 함수 내부에서 직접 causal mask를 생성해야 함.

```python
# 하삼각 행렬로 causal mask 생성
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
causal_mask = (1.0 - causal_mask) * -10000.0
scores = scores + causal_mask  # 상삼각(미래) 위치를 -10000으로 마스킹
```

처음에 causal mask 없이 구현했다가 `sanity_check.py` 실패 → causal mask 추가 후 통과.

### 2.3 `modules/gpt2_layer.py` — GPT2Layer

**핵심 포인트: Pre-LN 구조**

GPT-2는 BERT와 달리 **Pre-LN** 구조 사용 (Attention/FFN 이전에 LayerNorm 적용)

```
x → LayerNorm → CausalSelfAttention → dense → dropout → residual → x₁
x₁ → LayerNorm → Linear(GELU) → Linear → dropout → residual → 출력
```

`add()` 헬퍼: `return input + dropout(dense_layer(output))`  
(LayerNorm은 add()에서 적용하지 않고 forward()에서 처리)

### 2.4 `models/gpt2.py` — GPT2Model

**`embed()`**: 토큰 임베딩 + 위치 임베딩 더한 후 dropout 적용

**`hidden_state_to_token()`**: Weight Tying — 입력 임베딩 가중치 전치하여 logit 계산  
```python
return hidden_state @ self.word_embedding.weight.T
```

### 2.5 `classifier.py` — GPT2SentimentClassifier

GPT-2 마지막 토큰의 hidden state를 분류 헤드에 통과시켜 감정 예측

```python
# __init__()
self.dropout = nn.Dropout(config.hidden_dropout_prob)
self.classifier = nn.Linear(config.hidden_size, self.num_labels)

# forward()
last_token = self.gpt(input_ids, attention_mask)['last_token']
return self.classifier(self.dropout(last_token))
```

---

## 3. 실험 결과

### 3.1 last-linear-layer 모드 (GPT-2 가중치 고정)

| 데이터셋 | Dev Accuracy | 목표 | 상태 |
|----------|-------------|------|------|
| SST | 0.453 | 0.462 | 🟡 근접 |
| CFIMDB | 0.878 | 0.861 | ✅ 초과 달성 |

실행 명령어:
```bash
python classifier.py --fine-tune-mode last-linear-layer --use_gpu
```

### 3.2 full-model 모드 (GPT-2 전체 fine-tuning)

| 데이터셋 | Dev Accuracy | 목표 |
|----------|-------------|------|
| SST | - | 0.513 |
| CFIMDB | - | 0.976 |

> ⚠️ **주의**: `--lr 1e-5` 사용 필수!  
> 기본값 `lr=1e-3`은 full-model에 너무 커서 사전학습 가중치가 손상됨 (catastrophic forgetting).  
> classifier.py의 --lr 도움말에도 "finetune: 1e-5" 명시되어 있음.

권장 실행 명령어:
```bash
python classifier.py --fine-tune-mode full-model --lr 1e-5 --use_gpu
```

---

## 4. 알려진 이슈

### 4.1 PyTorch 2.6 호환성 (Colab 환경)

Colab은 PyTorch 2.6을 사용하며, `torch.load` 기본값이 `weights_only=True`로 변경됨.  
코드 수정 없이 해결하려면 실행 전 아래 코드를 셀에 추가:

```python
import torch, types
torch.serialization.add_safe_globals([types.SimpleNamespace])
```

### 4.2 h.{0...11}.attn.bias UNEXPECTED 경고

```
h.{0...11}.attn.bias | UNEXPECTED
```

이 경고는 무시해도 됨. GPT-2의 causal mask bias가 우리 구현과 아키텍처가 달라서 생기는 것으로,  
실제 동작에는 영향 없음 (`sanity_check.py` 통과로 확인됨).

---

## 5. 실행 환경

- conda 환경: `nlp_final` (`env.yml` 기반)
- GPU 필요 (Colab T4 GPU 권장)
- 로컬 Python: `C:\Users\jys72\anaconda3\envs\nlp_final\python.exe`

---

## 6. 제출 방법

8개 prediction 파일 생성 확인 후:
```bash
python prepare_submit.py
```
→ `nlp2026-final-outputs.zip` 생성 → 이클래스 "지정주제 프로젝트(기준모델)" 제출

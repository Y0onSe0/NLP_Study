# PART-II Paraphrase Detection 실험 계획

## 실행 순서

1. Tokenizer verbalizer를 확인한다: prompt는 trailing space로 끝나며, 다음 token은 unspaced `yes` / `no`를 사용한다.
2. `baseline`, `direct`, `meaning` 세 prompt만 subset + 1 epoch로 screening한다.
3. 가장 좋은 prompt 하나를 골라 full training checkpoint를 만든다.
4. 같은 checkpoint로 단방향 dev prediction과 bidirectional dev prediction을 비교한다.
5. dev set에서만 threshold를 `0.30~0.70`, step `0.01`로 calibration한다.
6. 확정된 checkpoint, prompt, bidirectional 여부, dev-selected threshold로 dev error analysis를 생성한다.
7. 모든 설정이 확정된 뒤 `test_predict`를 한 번만 실행해 최종 test prediction을 만든다.

## Test Set 사용 원칙

- 개발, prompt 선택, threshold 선택, error analysis에는 train/dev만 사용한다.
- test set은 `test_predict` 모드에서 최종 제출 파일을 만들 때만 읽는다.
- test prediction을 확인한 뒤 checkpoint, prompt, threshold를 다시 바꾸지 않는다.

## 보고서 Ablation Table 기준

1. Baseline
2. Prompt 개선
3. Prompt 개선 + Bidirectional inference
4. Prompt 개선 + Bidirectional inference + Threshold calibration

각 행에는 checkpoint, prompt template, bidirectional 여부, threshold, dev accuracy, dev macro-F1, prediction 파일명을 함께 기록한다.

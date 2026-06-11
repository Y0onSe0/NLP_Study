# PART-II Paraphrase Detection 실험 계획

## 실행 순서

1. Tokenizer verbalizer를 확인한다: prompt는 trailing space로 끝나며, 다음 token은 unspaced `yes` / `no`를 사용한다.
2. `baseline`, `direct`, `meaning` 세 prompt를 train 5,000개, dev 1,000개, 1 epoch 조건으로 screening한다.
3. 가장 좋은 `direct` prompt로 full training checkpoint를 만든다.
4. 같은 checkpoint로 단방향 dev prediction과 bidirectional dev prediction을 비교한다.
5. dev set에서만 threshold를 `0.30~0.70`, step `0.01`로 calibration한다.
6. 확정된 checkpoint, prompt, bidirectional 여부, dev-selected threshold로 dev error analysis를 생성한다.
7. 모든 설정이 확정된 뒤 `test_predict`를 한 번만 실행해 최종 test prediction을 만든다.

## Test Set 사용 원칙

- 개발, prompt 선택, threshold 선택, error analysis에는 train/dev만 사용한다.
- test set은 `test_predict` 모드에서 최종 제출 파일을 만들 때만 읽는다.
- test prediction을 확인한 뒤 checkpoint, prompt, threshold를 다시 바꾸지 않는다.

## Split 표기 검증 메모

`data/quora-dev.csv`를 읽은 실행은 dev 사용으로 기록한다. 과거 로그에 `Loaded ... train examples from data/quora-dev.csv`처럼 표시된 문구는 dev 파일을 train split으로 개발했다는 뜻이 아니라, `load_paraphrase_data()`의 `split` 기본값이 `train`이라 출력 라벨만 잘못 찍힌 것이다. 이 혼동을 막기 위해 dev 로드 호출은 `split='dev'`를 명시하도록 수정했다.

## 보고서 Ablation Table 기준

1. Prompt screening baseline/direct/meaning
2. Full direct
3. Full direct + Bidirectional inference
4. Full direct + Bidirectional inference + Threshold calibration

각 행에는 checkpoint, prompt template, bidirectional 여부, threshold, dev accuracy, dev macro-F1, prediction 파일명을 함께 기록한다. 보고서 표에는 `results/report_summary_20260611.csv`를 우선 사용한다.

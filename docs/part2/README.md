# PART-II Paraphrase Detection 문서

이 폴더는 자연어처리 기말 프로젝트 PART-II에서 선택한 Quora Paraphrase Detection 태스크의 구현 구조, 성능향상 방법론, 실행 절차를 정리한다. 본 프로젝트에서는 원래의 이진 분류 문제를 GPT-2의 next-token prediction 구조에 맞게 Cloze-style yes/no 생성 문제로 재구성한다. 모델은 두 질문을 포함한 prompt의 마지막 위치에서 `yes` 또는 `no` token 확률을 비교해 paraphrase 여부를 예측한다.

기존 실험 계획 요약은 [../PART2_실험계획.md](../PART2_실험계획.md)에 있으며, 이 폴더의 문서들은 그 계획을 보고서와 구현 가이드 형태로 확장한다.

## 문서 구성

- [01_baseline_implementation.md](01_baseline_implementation.md): baseline 구현 설명
- [02_improvement_methodology.md](02_improvement_methodology.md): 성능향상 방법론 설명
- [03_implementation_guide.md](03_implementation_guide.md): 실제 실행 순서와 구현/검증 방법

## 전체 실험 흐름

1. Baseline 신뢰성 확인
2. Prompt template screening
3. Full training
4. Bidirectional inference 적용
5. Dev threshold calibration
6. Dev error analysis
7. 최종 test prediction 생성

## Test set 사용 원칙

- 개발, prompt 선택, threshold 선택, error analysis에는 train/dev만 사용한다.
- test set은 `test_predict` 모드에서 최종 제출 파일 생성 시에만 사용한다.
- test prediction을 확인한 뒤 checkpoint, prompt, threshold를 다시 바꾸지 않는다.

## 최종 ablation table 기준

- Baseline
- Prompt 개선
- Prompt 개선 + Bidirectional inference
- Prompt 개선 + Bidirectional inference + Threshold calibration

## 보고서용 요약 문단

PART-II Paraphrase Detection은 Quora question pair의 두 질문이 같은 의미인지 판단하는 태스크이다. 본 프로젝트에서는 이 문제를 별도의 분류 head 중심 구조가 아니라 GPT-2의 next-token prediction에 맞춘 Cloze-style 문제로 구성한다. 입력 prompt는 두 질문과 yes/no 답변 지시문으로 구성되며, 모델은 마지막 token 위치에서 `yes`와 `no`의 확률을 비교한다. 개발 과정에서는 train/dev만 사용해 prompt, threshold, inference 방식을 선택하고, test set은 최종 제출 파일을 생성할 때 한 번만 사용한다. 최종 보고서는 baseline부터 prompt 개선, bidirectional inference, threshold calibration까지 단계별 ablation으로 성능 변화를 설명한다.

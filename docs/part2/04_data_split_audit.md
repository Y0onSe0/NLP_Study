# Data Split 사용 검증

## 결론

PART-II Paraphrase Detection 개발 과정에서 prompt 선택, checkpoint 선택, bidirectional inference 비교, threshold calibration, error analysis는 test set이 아니라 train/dev 기준으로 수행했다. Test set은 최종 제출용 prediction 파일을 생성하는 `test_predict` 모드에서만 읽는다.

## 코드 기준 사용 내역

| 단계 | 실행 모드 | 읽는 파일 | 용도 |
|---|---|---|---|
| 학습 | `train_dev` | `data/quora-train.csv` | 모델 파라미터 업데이트 |
| checkpoint 선택 | `train_dev` | `data/quora-dev.csv` | epoch별 dev accuracy 평가 및 best checkpoint 저장 |
| prompt/inference 비교 | `dev_predict` | `data/quora-dev.csv` | dev accuracy, dev macro-F1 비교 |
| threshold 선택 | `calibrate_dev` | `data/quora-dev.csv` | `p_yes` 기준 threshold 탐색 |
| 오류 분석 | `error_analysis` | `data/quora-dev.csv` | false positive, false negative, borderline case 분석 |
| 최종 제출 예측 | `test_predict` | `data/quora-test-student.csv` | label 없는 test prediction 생성 |

## 로깅 표기 수정

수정 전 `logs/full-direct.log`에는 `Loaded 40429 train examples from data/quora-dev.csv`라는 문구가 있었다. 이는 `load_paraphrase_data()` 호출에서 dev 파일을 읽으면서 `split` 인자를 명시하지 않아 기본값 `train`이 출력된 로깅 표기 오류다. 실제 파일 경로는 `data/quora-dev.csv`였고, dev 파일은 label이 있는 평가용 split으로 사용되었다.

이 혼동을 막기 위해 `paraphrase_detection.py`의 dev 로드 지점은 모두 `load_paraphrase_data(args.para_dev, split='dev')`로 수정했다. 기존 로그의 해당 두 줄도 파일 경로에 맞게 `Loaded 40429 dev examples from data/quora-dev.csv`로 바로잡았다.

## 재실험 필요성 판단

현재 확인된 문제는 데이터 파일 선택 오류가 아니라 split 이름 출력 오류다. 따라서 기존 full training 결과를 폐기하고 다시 학습할 필요는 없다. 다만 만약 별도 실행 기록에서 `data/quora-test-student.csv`를 사용한 뒤 prompt, checkpoint, threshold를 다시 바꾼 흔적이 발견되면 그 경우에는 test leakage로 보고 train/dev 기준으로 다시 선택해야 한다.

현재 `results/paraphrase_experiments.csv` 기준으로 test set을 읽은 기록은 `full-direct-final,test_predict` 한 행이며, 이 행에는 dev accuracy/F1이나 selected threshold가 기록되어 있지 않다. 이는 최종 prediction 생성 용도였음을 의미한다.

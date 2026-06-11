# Results Organized by Mode

이 폴더는 원본 산출물 경로를 유지하면서, 사람이 확인하기 쉽도록 PART-II Paraphrase Detection 결과를 모드별로 모아 둔 인덱스다. 제출 스크립트와 기존 문서가 참조하는 원본 파일은 옮기지 않았다.

## Folder Mapping

| folder | source mode | contents |
|---|---|---|
| `train/` | `train_dev` | 학습 기록, checkpoint 링크, train/dev 평가 실험 행 |
| `dev/` | `dev_predict` | dev prediction 관련 실험 행과 남아 있는 dev prediction 링크 |
| `calibrate_dev/` | `calibrate_dev` | dev threshold calibration 실험 행 |
| `error_analysis/` | `error_analysis` | dev error analysis 실험 행과 error analysis CSV 링크 |
| `test/` | `test_predict` | 최종 test prediction 실험 행과 제출 prediction 링크 |

각 폴더의 `experiments.csv`는 `results/paraphrase_experiments.csv`에서 해당 mode 행만 필터링한 파일이다. `artifact_status.tsv`는 실험 로그에 기록된 checkpoint/prediction 경로가 현재 디스크에 남아 있는지 표시한다.

## Important

이 구조는 정리용 복사/링크 모음이다. 실제 실행 기본 경로는 여전히 `checkpoints/`, `predictions/`, `logs/`, `results/` 아래 원본 파일이다.

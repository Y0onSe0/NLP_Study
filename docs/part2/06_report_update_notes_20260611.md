# Report Update Notes (2026-06-11)

이 문서는 코드/문서 정리 이후 최종 보고서에 반영해야 할 변경사항만 요약한다.

## 결과 숫자

보고서의 핵심 PART-II 숫자는 유지한다. 단, 표를 작성할 때는 raw 실행 이력 CSV가 아니라 아래 파생 요약 CSV를 기준으로 삼는다.

- Report summary: `results/report_summary_20260611.csv`
- Raw rerun history: `results/rerun-20260611/paraphrase_experiments.csv`

`results/report_summary_20260611.csv`는 raw rerun CSV에서 보고서에 필요한 행만 추출한다. Raw CSV에는 실행 이력 보존을 위해 중복 row가 남아 있을 수 있으므로, 보고서 표에는 summary CSV를 사용하는 편이 안전하다.

## 재현 프로토콜

보고서의 실험 환경 또는 재현 절차에는 다음 기준을 명시한다.

- Canonical rerun script: `scripts/rerun_part2_report_20260611.sh`
- Canonical protocol doc: `docs/part2/05_report_rerun_protocol_20260611.md`
- Prompt screening: train 5,000 / dev 1,000, 1 epoch
- Full training: full train set, direct prompt, 3 epochs
- Training batch size 4, gradient accumulation 2
- Max length 128
- Bidirectional inference and threshold calibration are dev-only
- Test set is used only by `test_predict`

## Colab 관련 문구

`PART2_Colab_Run.ipynb`는 legacy helper로 남긴다. 보고서에는 Colab을 canonical 실행 환경으로 쓰지 않는다. Colab 결과를 보고서에 쓰려면 notebook command sequence, hyperparameters, checkpoint name, split usage가 GCP rerun protocol과 일치해야 한다.

## CSV 해석 주의

새 코드에서는 앞으로 evaluation/test row에 학습용 hyperparameter나 생성하지 않는 prediction path가 기록되지 않도록 `record_experiment()`를 수정했다. 기존 raw CSV는 실행 당시의 원본 이력으로 유지한다.

보고서에는 다음처럼 표현하는 것이 안전하다.

- "Raw execution history is stored in `results/rerun-20260611/paraphrase_experiments.csv`."
- "The report tables use the cleaned derived summary in `results/report_summary_20260611.csv`."
- "Logs were used only for execution tracing, not as final result artifacts."

## 산출물 표 업데이트

최종 산출물 표에는 다음 파일을 추가한다.

| 항목 | 파일 |
|---|---|
| 보고서용 요약 CSV | `results/report_summary_20260611.csv` |
| rerun raw experiment CSV | `results/rerun-20260611/paraphrase_experiments.csv` |
| rerun error analysis CSV | `results/rerun-20260611/error_analysis_para.csv` |
| rerun final test prediction | `predictions/rerun-20260611/para-test-final.csv` |

기존 `predictions/para-test-final.csv`와 `results/paraphrase_experiments.csv`는 원본 실행 산출물로 남긴다. 보고서 재현 검증에는 `rerun-20260611` 경로와 summary CSV를 우선 사용한다.

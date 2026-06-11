# 자연어처리 2026-1 지정주제 기말 프로젝트: GPT-2 구축

## PART-I:

#### 다음 각 모듈에서 누락된 코드 블록을 완성해야 한다.
* `modules/attention.py`
* `modules/gpt2_layer.py`
* `models/gpt2.py`
* `classifier.py`
* `optimizer.py`

#### 다음 모듈들을 실행하여 PART-I의 구현을 테스트한다.

* `optimizer_test.py`: `optimizer.py` 구현을 테스트.
* `sanity_check.py`: GPT 모델 구현을 테스트.
* `classifier.py`: 모델을 사용한 감정 분류 수행.

## PART-II

#### 다음 모듈들을 실행하여 PART-II의 구현을 테스트한다.

* `paraphrase_detection.py`: 패러프레이즈 탐지 수행.
* `sonnet_generation.py`: 소네트 생성 수행.

**주목**: 사용하는 GPU 사양에 따라 batch_size 같은 하이퍼파라미터를 조정하여 성능을 최적화하고 메모리 부족 오류를 방지해야 한다.

#### PART-II 테스트의 핵심 포인트

두 파일에 있는 누락된 코드 블록을 완성하는 것도 중요하지만, PART-II의 핵심은 기능의 확장에 있다. GPT-2 모델을 수정하여 한 문장이 다른 문장의 패러프레이즈인지 판단하는 능력과 소네트를 생성하는 능력을 개선하는 방법에 촛점을 맞추도록 하자.

## 제출 전 확인

### Part-I 필수 파일 및 prediction 확인

```bash
python -m py_compile modules/attention.py modules/gpt2_layer.py models/base_gpt.py models/gpt2.py classifier.py optimizer.py paraphrase_detection.py datasets.py evaluation.py
python prepare_submit.py
ls -lh nlp2026-final-outputs.zip
unzip -l nlp2026-final-outputs.zip | head -50
```

Part-I SST/CFIMDB prediction 파일은 `predictions/` 아래에 있어야 하며, 각 CSV는 `id, Predicted_Sentiment` 형식의 header와 id별 예측 label을 포함해야 한다.

### Part-II Paraphrase Detection 확인

```bash
python paraphrase_detection.py --help
```

`paraphrase_detection.py`의 실행 모드는 `train_dev`, `dev_predict`, `calibrate_dev`, `error_analysis`, `test_predict`이다. Prompt 선택, threshold calibration, error analysis는 train/dev 기준으로만 수행하고, test set은 checkpoint, prompt, threshold가 모두 확정된 뒤 최종 prediction 생성에만 사용한다. Split 사용 검증과 기존 로그 표기 오류 설명은 `docs/part2/04_data_split_audit.md`에 정리했다.

### 데이터셋 준비

현재 실행에 필요한 소규모 과제 데이터는 `data/` 폴더에 포함되어 있다. 평가자는 별도 다운로드 없이 기본 경로의 `data/*.csv`와 `data/*.txt`를 사용해 실행할 수 있어야 한다.

## 환경 설정
**주목**: .yml 파일의 버전을 변경하지 말것.

#### GitHub에서 Source code 내려 받기:
* 단순히 압축 파일 내려받아서 풀지 말고 GitHub의 프로젝트 리포지토리를 클론할 것.
* 프로젝트 폴더를 만들 폴더로 가서 다음 명령을 터미널에서 실행한다.
```
git clone https://github.com/kikim6114/nlp2026-fina1.git
```
* 소스코드 변경 사항이 있을 경우 공지가 나가므로, 그 경우 `git pull` 하여 PC의 로컬 리포지토리를 업데이트할 수 있다.

#### 파이썬 설치
* anaconda3 를 설치한다.

#### 환경 및 패키지 설치

* conda env create -f env.yml
* conda activate nlp_final  

**주의**:
* 프로젝트 PART-I을 수행하면서, 위에서 설치된 패키지만을 사용해야 하며, 별도의 다른 패키지는 허용되지 않는다.
* 모든 command 옵션이나 파라미터는 변경/추가하면 안된다.



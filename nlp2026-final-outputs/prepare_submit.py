# 과제 평가를 위해 zip 파일을 만들어 제출합니다.

import os
import zipfile

def listed_files(path):
    return [os.path.join(path, p) for p in os.listdir(path)
            if os.path.isfile(os.path.join(path, p))]

required_prediction_files = [
    'predictions/last-linear-layer-sst-dev-out.csv',
    'predictions/last-linear-layer-sst-test-out.csv',
    'predictions/full-model-sst-dev-out.csv',
    'predictions/full-model-sst-test-out.csv',
    'predictions/last-linear-layer-cfimdb-dev-out.csv',
    'predictions/last-linear-layer-cfimdb-test-out.csv',
    'predictions/full-model-cfimdb-dev-out.csv',
    'predictions/full-model-cfimdb-test-out.csv',
]

required_files = [p for p in os.listdir('.') if p.endswith('.py')] + \
                 required_prediction_files + \
                     listed_files('models') + \
                        listed_files('modules')

def main():
    aid = 'nlp2026-final-outputs'

    with zipfile.ZipFile(f"{aid}.zip", 'w') as zz:
        for file in required_files:
            zz.write(file, os.path.join(".", file))
    print(f"Submission zip file created: {aid}.zip")

if __name__ == '__main__':
    main()

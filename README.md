# Time Series Explainability

**시계열 분류 모델의 예측을 LIME, SHAP, Integrated Gradients로 분석한 팀 프로젝트입니다.**
Wafer와 FingerMovements 데이터를 대상으로 데이터 처리, PyTorch 모델 학습, 성능 평가와 예측 근거 시각화를 수행했습니다.
이 저장소는 과제의 연결된 학습·설명 코드와 기존 실행 기록을 심사자가 읽기 쉽도록 정리한 포트폴리오입니다.

| 구분 | 내용 |
|---|---|
| 문제 | 제조 공정 센서 및 EEG 시계열의 이진 분류와 예측 해석 |
| 구현 | Python, PyTorch, sktime, scikit-learn |
| 설명·시각화 | LIME, SHAP, Captum, Matplotlib, Seaborn |
| 프로젝트 형태 | 설명가능한 기계학습 수업 팀 프로젝트 |

## Workflow

`.ts` 데이터 로딩 → 배열·라벨 변환 → Transformer 학습 → 분류 지표·혼동행렬 → 세 설명 기법 비교

## Repository

```text
src/
├── train_explain.py       # 데이터 로딩, 모델 학습·평가, 설명 호출
└── explainability.py      # LIME, SHAP, Integrated Gradients
data/README.md             # 원 데이터 출처와 폴더 구성
results/
├── historical_run.txt     # 기존 실행 로그
└── historical_metrics.csv # 로그에서 옮긴 요약 지표
docs/experiment-notes.md   # 실험 출처와 해석상의 한계
requirements.txt
provenance.json
```

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

[데이터 준비 안내](data/README.md)에 따라 파일을 배치한 후 저장소 루트에서 실행합니다.

```bash
# 먼저 분류 모델 학습·평가만 실행
python src/train_explain.py --dataset Wafer --epochs 10 --explanations none

# 두 데이터셋에 세 설명 기법 적용
python src/train_explain.py --dataset both --epochs 20 --explanations all
```

그래프는 Matplotlib 창에 표시됩니다. CPU 또는 사용 가능한 CUDA 장치를 선택합니다.
의존성에는 원 로컬 환경에서 확인한 버전을 기록했지만 SHAP·Captum 버전은 복원되지 않았습니다.
전체 학습·설명 재실행은 게시 과정에서 수행하지 않았습니다.

## Recorded results

아래는 **기존 로그의 수치**이며 이번에 새로 실행한 결과가 아닙니다.

| Dataset | Accuracy | F1 of class 1 |
|---|---:|---:|
| Wafer | 0.9857 | 0.9920 |
| FingerMovements | 0.5500 | 0.6457 |

데이터셋별 난이도에 따른 성능 차이가 관찰됩니다. 원 코드의 배열 축과 설명 피처 처리에는 점검할 부분이 있어,
이 수치를 개선된 시계열 모델이나 설명 기법의 정확성 검증 결과로 제시하지 않습니다.
자세한 범위는 [실험 기록과 한계](docs/experiment-notes.md)를 참고하세요.

## Portfolio evidence

- **데이터 처리 언어 활용:** Python 배열 처리, 라벨 매핑, 학습 데이터 구성
- **머신러닝 개발 결과물:** 모델과 학습·평가·설명 코드를 하나의 저장소로 구성
- **데이터 시각화:** 혼동행렬, 설명 피처 중요도, 기여도 시각화 코드

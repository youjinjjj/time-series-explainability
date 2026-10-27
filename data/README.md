# Dataset preparation

이 프로젝트는 UCR/UEA Time Series Classification Archive의 두 데이터셋을 사용합니다.

- [Wafer](https://www.timeseriesclassification.com/description.php?Dataset=Wafer): 제조 공정 센서 기반 이진 분류
- [FingerMovements](https://www.timeseriesclassification.com/description.php?Dataset=FingerMovements): 다변량 EEG 기반 이진 분류

원 출처에서 `.ts` 형식의 학습·시험 파일을 준비하고, 다음 구조로 배치하세요.

```text
data/
├── Wafer/
│   ├── Wafer_TRAIN.ts
│   └── Wafer_TEST.ts
└── FingerMovements/
    ├── FingerMovements_TRAIN.ts
    └── FingerMovements_TEST.ts
```

데이터 자체와 학습 가중치는 저장소에 포함하지 않습니다. 데이터 이용 조건은 각 원 출처를 따릅니다.
별도 경로를 사용할 때는 `--data-dir` 옵션을 지정하세요.

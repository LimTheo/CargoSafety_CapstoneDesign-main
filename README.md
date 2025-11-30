#  CapstoneDesign

### 고려사항
1. Python 버전 통합을 해야하는지 실행해봐야 할듯
2. 변수 통합 하지 않고 각각의 기능들을 메인에서 불러와 돌렸을때 기능의 오류가 없을지
3. 화물검출시에 기존 이미지 -> 실시간 처리 코드 변경 시 라즈베리파이 에서 문제없이 검출이 될지

### 현재 파일 구조
~~~text
.
├── CargoSafety_CapstoneDesign/
├── ├── src/                  # 주요 소스 코드
├── │   ├── detection/        # 화물 검출 기능
├── │   │   ├── object_detection.py  # YOLOE 예측, 필터링, 데이터셋 생성
├── │   │   └── masking.py    # 배경 마스킹 (분리)
├── │   ├── tilt/             # 기울기 계산 기능
├── │   │   └── tilt_detection.py  # detect_pallet_tilt_with_graph 함수 등
├── │   ├── common/           # 공통
├── │   │   ├── camera_input.py  # 실시간 카메라
├── │   │   └── visualization.py  # imshow, plt.show 등
├── │   └── models/           # 모델 로더
├── │       └── yoloe_loader.py  # YOLOE 로드 & 클래스
├── ├── data/                 # 데이터
├── │   ├── yoloe-v8l-seg.pt  # hf_hub_download로 다운로드
├── │   ├── yoloe-v8l-seg-pf.pt
├── │   ├── ram_tag_list.txt  # wget
├── ├── tests/                # 테스트 (예시만)
├── │   └── test_tilt.py      # tilt 함수 테스트
├── ├── requirements.txt      # 의존성
├── ├── main.py               # 메인: 실시간 실행, 두 기능 연결
└── └── README.md             # 설명
~~~
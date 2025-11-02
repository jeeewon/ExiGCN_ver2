ExiGCN/
├── README.md
├── requirements.txt
├── setup.py
│
├── config/
│   ├── __init__.py
│   ├── base_config.yaml          # 기본 하이퍼파라미터
│   ├── cora_full.yaml            # Cora-Full 설정
│   ├── amazon_computers.yaml     # Amazon-Computers 설정
│   ├── ogbn_arxiv.yaml           # OGBN-Arxiv 설정
│   └── reddit.yaml               # Reddit 설정
│
├── data/
│   ├── __init__.py
│   ├── download.py               # 데이터셋 다운로드
│   ├── preprocessor.py           # 전처리 (정규화, split 등)
│   └── graph_updater.py          # 그래프 변경 시뮬레이션 (incremental/deletion)
│
├── models/
│   ├── __init__.py
│   ├── base_gcn.py               # 기본 GCN 구현
│   ├── exigcn.py                 # ExiGCN 메인 모델
│   ├── twp.py                    # TWP 비교 모델
│   └── layers.py                 # GCN layer, activation 등
│
├── utils/
│   ├── __init__.py
│   ├── sparse_ops.py             # Sparse matrix (triplet) 연산
│   ├── cache_manager.py          # Z, H, F 캐싱 관리
│   ├── metrics.py                # Accuracy, F1 등 평가지표
│   ├── timer.py                  # 시간 측정 유틸
│   └── logger.py                 # 로깅 (wandb, tensorboard)
│
├── train/
│   ├── __init__.py
│   ├── trainer_full.py           # Full Retraining 학습
│   ├── trainer_exi.py            # ExiGCN 학습
│   ├── trainer_twp.py            # TWP 학습
│   └── optimizer.py              # Optimizer 설정
│
├── experiments/
│   ├── __init__.py
│   ├── run_incremental.py        # Incremental 실험
│   ├── run_deletion.py           # Deletion 실험
│   ├── run_comparison.py         # 전체 비교 실험
│   └── ablation_study.py         # Ablation study
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_test.ipynb
│   ├── 03_exigcn_test.ipynb
│   └── 04_result_analysis.ipynb
│
├── results/
│   ├── figures/                  # 그래프, 차트
│   ├── tables/                   # 실험 결과 테이블
│   └── checkpoints/              # 모델 체크포인트
│       ├── initial/              # 초기 90% 학습 결과
│       └── updated/              # 각 업데이트 단계별
│
├── tests/
│   ├── __init__.py
│   ├── test_sparse_ops.py
│   ├── test_exigcn.py
│   ├── test_graph_updater.py
│   └── test_equivalence.py       # Full vs ExiGCN 동등성 검증
│
└── scripts/
    ├── download_all_data.sh
    ├── run_all_experiments.sh
    └── generate_paper_plots.py
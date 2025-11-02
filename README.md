# ExiGCN: Exact and Efficient Graph Convolutional Networks

PyTorch implementation of ExiGCN from the paper "ExiGCN: Exact and Efficient Graph Convolutional Networks under Structural Dynamics" (BigComp 2026).

## Features

- ✅ Exact full retraining with significantly reduced computation
- ✅ 3-5× speedup on benchmark datasets
- ✅ Support for both node/edge addition and deletion
- ✅ GPU acceleration with PyTorch
- ✅ Stratified sampling for fair evaluation

## Project Structure

```
ExiGCN/
├── config/          # Configuration files for each dataset
├── data/            # Data loading and preprocessing
├── models/          # GCN and ExiGCN implementations
├── utils/           # Sparse operations, caching, metrics
├── train/           # Training logic
├── experiments/     # Experiment scripts
├── results/         # Outputs and checkpoints
└── tests/           # Unit tests
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Experiments

```bash
# Incremental scenario on Cora-Full
python experiments/run_incremental.py --dataset cora --config config/cora_full.yaml

# Full comparison (all datasets, all scenarios)
python experiments/run_comparison.py
```

## Datasets

- Cora-Full (19K nodes, citation network)
- Amazon-Computers (13K nodes, co-purchase)
- OGBN-Arxiv (169K nodes, citation network)
- Reddit (232K nodes, social network)

## Results

Results will be saved in `results/` directory with:
- Performance tables (CSV)
- Training time comparisons
- Speedup plots
- Model checkpoints

## Citation

```bibtex
@inproceedings{exigcn2026,
  title={ExiGCN: Exact and Efficient Graph Convolutional Networks under Structural Dynamics},
  booktitle={BigComp 2026},
  year={2026}
}
```
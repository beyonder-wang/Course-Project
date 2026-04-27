# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SUSTech ML & MEA course project — EEG signal classification across 5 datasets using PyTorch. The goal is to train models that predict test-set labels with maximal accuracy.

## Datasets (in `data/`)

| Dataset    | Task                  | Classes | Channels | Window | Files                     |
|------------|-----------------------|---------|----------|--------|---------------------------|
| MDD        | Depression diagnosis  | 2       | 20       | 1.0s   | `train.h5`, `val.h5`, `test_x_only.h5` |
| BCIC2A     | Motor imagery         | 4       | 22       | 1.0s   | same pattern              |
| CHINESE    | Reading detection     | 2       | 22       | 1.0s   | same pattern              |
| SEED       | Emotion recognition   | 3       | 62       | 2.0s   | same pattern              |
| SLEEP      | Sleep staging         | 5       | 6        | 30.0s  | same pattern              |

All preprocessed to 200 Hz, bandpass filtered 0.1–75 Hz. HDF5 files contain `X` (float32, shape N×C×T) and `y` (int64 labels). `test_x_only.h5` has only `X`.

After running `prepare_folds.py`, each dataset gains `all.h5` (merged train+val), `folds_info.json`, and `fold_{k}/train_idx.npy` + `val_idx.npy`.

## Available Models

Defined in `model/` package:
- **SimpleLinear** (`model/simple.py`) — flattened input → single linear layer (baseline)
- **SimpleMLP** (`model/simple.py`) — flattened input → hidden layers with ReLU+Dropout
- **EEGNet** (`model/eegnet.py`) — CNN with temporal/spatial/separable convolutions
- **EEGGRU** (`model/rnn.py`) — bidirectional GRU
- **EEGLSTM** (`model/rnn.py`) — bidirectional LSTM

## Key Files

| File | Purpose |
|------|---------|
| `0_run_train.py` | **CLI entry point** for training and test prediction |
| `prepare_folds.py` | **Upstream**: merge train+val, generate stratified 5-fold CV splits |
| `trainer.py` | `Trainer` class: training loop, validation, metric logging, prediction saving |
| `utils.py` | `load_dataset_info()`, `create_dataloaders()` (supports original or fold split) |
| `model/` | Package with model definitions (`__init__.py` exports `MODEL_DICT`) |
| `data/TEST_DATASET.py` | PyTorch Dataset classes: `TrainDataset`, `FoldDataset`, `TestDataset` |
| `train.ipynb` | Original Jupyter pipeline (deprecated in favor of CLI scripts) |
| `RNN_Exercise.py` | Legacy RNN/LSTM definitions (superseded by `model/rnn.py`) |
| `Results/{tag}/` | Per-run output: predictions.txt, model.pt, config.json, metrics.json |
| `tmp_results.md` | Log of model accuracy comparisons |

## Quick Commands (run from repo root)

```bash
# 1. Prepare CV folds (once per dataset)
python prepare_folds.py --dataset MDD

# 2. Train and predict
python 0_run_train.py --dataset MDD --model EEGNet --epochs 30 --lr 1e-3

# 3. Single fold training
python 0_run_train.py --dataset MDD --model EEGNet --epochs 30 --fold 1

# 4. Auto-run all 5 folds with CV summary
python 0_run_train.py --dataset MDD --model EEGNet --epochs 30 --fold -1

# View dataset info
python -c "import json; info=json.load(open('data/MDD/dataset_info.json')); print(info['dataset']['category_list'], len(info['dataset']['channels']))"

# Check saved predictions
cat Results/MDD*/predictions.txt | head -20
```

## `0_run_train.py` Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | MDD | Dataset name (MDD/BCIC2A/CHINESE/SEED/SLEEP) |
| `--model` | EEGNet | Model architecture |
| `--lr` | 5e-4 | Learning rate |
| `--epochs` | 5 | Number of epochs |
| `--batch_size` | 32 | Batch size |
| `--fold` | (none) | CV fold 1-5, or -1 for all 5 folds |

Each run creates an isolated `Results/{Dataset}_{Model}_{timestamp}/` dir:
```
Results/MDD_EEGNet_20260427_153025/
├── config.json         # run configuration
├── metrics.json        # final/best val accuracy + loss history
├── predictions.txt     # test set predictions (one label per line)
└── model.pt            # trained model state_dict
```

With `--fold -1`:
```
Results/MDD_EEGNet_20260427_..._allfolds/
├── config.json
├── cv_summary.json     # per-fold metrics + mean ± std
├── fold_1/{...}
├── fold_2/{...}
├── fold_3/{...}
├── fold_4/{...}
└── fold_5/{...}
```

## Architecture Notes

- All models expect input shape `(B, C, T)` — batch × channels × time
- RNN models transpose internally to `(B, T, C)` and use final hidden state for classification
- Training loop runs on CPU (no CUDA setup currently)
- Each dataset has different channel counts and class counts — configured automatically via `dataset_info.json`
- `--fold` requires `prepare_folds.py` to have been run first for that dataset

## Known Results (on MDD dataset)

| Model | Val Accuracy |
|-------|-------------|
| SimpleMLP | 75.00% |
| EEGNet (lr=1e-3) | 79.69% |
| EEGGRU (lr=5e-4) | 80.16% |
| ExerciseEEGLSTM (lr=5e-4) | 81.25% |

# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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

## Available Models

Defined in `model/` package:
- **SimpleLinear** (`model/simple.py`) — flattened input → single linear layer (baseline)
- **SimpleMLP** (`model/simple.py`) — flattened input → hidden layers with ReLU+Dropout
- **EEGNet** (`model/eegnet.py`) — CNN with temporal/spatial/separable convolutions (standard EEG baseline)
- **EEGGRU** (`model/rnn.py`) — bidirectional GRU
- **EEGLSTM** (`model/rnn.py`) — bidirectional LSTM

### Pre-training & MoE modules (in `model/`)
- **SimCLREncoder** (`model/simclr_model.py`) — LSTM encoder + projection head for contrastive learning
- **MoESimCLREncoder** (`model/simclr_model.py`) — SimCLREncoder with Mixture of Experts layer
- **MoELayer** (`model/moe.py`) — Top-k gated MoE with load balancing loss
- **ChannelAdapter** (`model/channel_adapter.py`) — Per-dataset 1×1 Conv1d to unify channel counts (Phase 2)
- **NTXentLoss** (`model/contrastive_loss.py`) — SimCLR normalized temperature-scaled cross-entropy
- **Augmentations** (`model/augmentations.py`) — GaussianNoise, ChannelDropout, TimeShift, Compose, SimCLRTransform

## Key Files

| File | Purpose |
|------|---------|
| `0_run_train.py` | **CLI entry point** for supervised training and test prediction |
| `1_run_pretrain.py` | **CLI entry point** for SimCLR pre-training + fine-tuning (Phases 1–2, supports MoE) |
| `2_run_benchmark.py` | **CLI entry point** for multi-dataset benchmark (baseline / finetune / compare) |
| `prepare_folds.py` | Merge train+val, generate stratified 5-fold CV splits |
| `trainer.py` | `Trainer` class: training loop, early stopping, device support, prediction saving |
| `pretrainer.py` | `Pretrainer` class: SimCLR contrastive training loop with augmentation + balance loss |
| `utils.py` | Data loaders, device resolution, log/summary helpers |
| `model/` | Package with model definitions (`__init__.py` exports `MODEL_DICT`) |
| `data/TEST_DATASET.py` | PyTorch Dataset classes: `TrainDataset`, `FoldDataset`, `TestDataset`, `UnlabeledDataset` |
| `train.ipynb` | Original Jupyter pipeline (deprecated in favor of CLI scripts) |
| `RNN_Exercise.py` | Legacy RNN/LSTM definitions (superseded by `model/rnn.py`) |
| `Results/{tag}/` | Per-run output: predictions.txt, model.pt, config.json, metrics.json, summary.txt, run.log |
| `Pretrained/{tag}/` | Pre-trained encoder weights: encoder.pt, adapter.pt, config.json, pretrain_metrics.json |

## Training Pipeline

Use the CLI entry points instead of the notebook:

```bash
# === Supervised training ===
python 0_run_train.py --dataset MDD --model EEGLSTM --epochs 30 --lr 5e-4 --device cpu
python 0_run_train.py --dataset MDD --model EEGLSTM --epochs 30 --fold -1  # 5-fold CV

# === SimCLR pre-training + fine-tuning ===
# Phase 1: single-dataset pretrain + finetune
python 1_run_pretrain.py --phase 1 --action both --dataset MDD --epochs_pretrain 50 --epochs_finetune 30
# Phase 1 + MoE
python 1_run_pretrain.py --phase 1 --action both --dataset MDD --use_moe --moe_num_experts 4 --epochs_pretrain 50
# Phase 2: multi-dataset pretrain then finetune
python 1_run_pretrain.py --phase 2 --action pretrain --epochs_pretrain 50 --device cpu
python 1_run_pretrain.py --phase 2 --action finetune --dataset MDD --pretrained_encoder Pretrained/<tag>/encoder.pt

# === Benchmark ===
python 2_run_benchmark.py --mode baseline --model EEGLSTM --epochs 30 --device cpu
python 2_run_benchmark.py --mode compare
```

## Quick Commands (run from repo root)

```bash
# View dataset info
python -c "import json; info=json.load(open('data/MDD/dataset_info.json')); print(info['dataset']['category_list'], len(info['dataset']['channels']))"

# Check saved predictions
cat Results/MDD_*/predictions.txt | head -20
```

## Architecture Notes

- All models expect input shape `(B, C, T)` — batch × channels × time
- RNN models transpose internally to `(B, T, C)` and use final hidden state for classification
- Device selection via `--device` flag (cpu / cuda / cuda:N / auto); falls back to CPU if CUDA unavailable
- Each dataset has different channel counts and class counts — configured automatically via `dataset_info.json`
- `--fold` requires `prepare_folds.py` to have been run first for that dataset

## Known Results (on MDD dataset)

| Model | Val Accuracy |
|-------|-------------|
| SimpleMLP | 75.00% |
| EEGNet (lr=1e-3) | 79.69% |
| EEGGRU (lr=5e-4) | 80.16% |
| EEGLSTM (lr=5e-4) | 81.25% |
| SimCLR pretrain + finetune | **87.50%** |

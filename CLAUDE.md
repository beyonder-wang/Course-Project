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

## Important: Prediction Files Are Git-Tracked

Prediction `.txt` files (predictions.txt, MDD.txt, etc.) are **the most important output** and are tracked in git via `.gitignore` negation rules. These are force-added with `git add -f` when they live inside the `Results/` directory. New prediction files from training runs should always be committed.

## Available Models

Defined in `model/` package:
- **SimpleLinear** (`model/simple.py`) — flattened input → single linear layer (baseline)
- **SimpleMLP** (`model/simple.py`) — flattened input → hidden layers with ReLU+Dropout
- **EEGNet** (`model/eegnet.py`) — CNN with temporal/spatial/separable convolutions
- **EEGGRU** (`model/rnn.py`) — bidirectional GRU
- **EEGLSTM** (`model/rnn.py`) — bidirectional LSTM
- **EEGMamba** (`model/mamba_model.py`) — bidirectional Mamba (selective state-space model)

### KAN classifier variants (`model/kan_backbones.py`)
Replace the MLP classifier head with KANMLP. Add `_KAN` suffix:
- **EEGLSTM_KAN**, **EEGGRU_KAN**, **EEGNet_KAN**, **EEGMamba_KAN**, **SimpleMLP_KAN**

### Attention-enhanced variants (`model/attention_eeg.py`)
- **EEGNet_SE** — EEGNet + Squeeze-and-Excitation channel attention
- **EEGNet_SimAM** — EEGNet + parameter-free SimAM attention (zero extra params)
- **EEGNet_SimAM_SE** — EEGNet with both SE + SimAM attention

### Motor imagery specific models
- **FBCNet** (`model/fbcnet.py`) — Multi-band FFT filtering + depthwise spatial conv + variance pooling, ~4.5K params (Ravi et al. 2021)
- **EEG-TCNet** (`model/tcnet.py`) — EEGNet frontend + TCN backend, ~14.8K params (Ingolfsson et al. 2020)
- **ShallowConvNet** (`model/shallownet.py`) — FBCSP-inspired: temporal conv + spatial conv + square + pool + log, ~43.8K params (Schirrmeister et al. 2017)
- **ATCNet** (`model/atcnet.py`) — EEGNet conv frontend + sliding-window multi-head attention + causal TCN + ensemble, ~113.7K params (Altaheri et al. 2023, **best on BCIC2A: 66.20%**)

### Pre-training & MoE modules (in `model/`)
- **SimCLREncoder** (`model/simclr_model.py`) — LSTM encoder + projection head for contrastive learning; weight-transfer compatible with EEGLSTM
- **MoESimCLREncoder** (`model/simclr_model.py`) — SimCLREncoder with Mixture of Experts between LSTM and projection head
- **MultiBandSimCLREncoder** (`model/multiband_simclr.py`) — Multi-band SimCLR: FFT band decomposition + shared LSTM + per-band projection heads
- **MultiBandMoESimCLREncoder** (`model/multiband_simclr.py`) — Multi-band SimCLR with MoE between LSTM and projections
- **MoELayer** (`model/moe.py`) — Top-k gated MoE with load balancing loss
- **ChannelAdapter** (`model/channel_adapter.py`) — Per-dataset 1×1 Conv1d to unify channel counts (Phase 2)
- **NTXentLoss** (`model/contrastive_loss.py`) — SimCLR normalized temperature-scaled cross-entropy
- **MultiBandNTXentLoss** (`model/multiband_loss.py`) — Multi-head NT-Xent loss per frequency band
- **KANLinear / KANMLP** (`model/kan.py`) — Kolmogorov-Arnold Network layer (learnable B-spline activations)
- **BandDecomposition** (`model/band_decomposition.py`) — FFT-based EEG frequency band splitting (delta/theta/alpha/beta/gamma)
- **Augmentations** (`model/augmentations.py`) — GaussianNoise, ChannelDropout, TimeShift, Compose, SimCLRTransform

## Key Files

| File | Purpose |
|------|---------|
| `0_run_train.py` | **CLI entry point** for supervised training and test prediction |
| `1_run_pretrain.py` | **CLI entry point** for SimCLR pre-training + fine-tuning (Phases 1–4, supports MoE) |
| `2_run_benchmark.py` | **CLI entry point** for multi-dataset benchmark (baseline / finetune / compare) |
| `3_run_multiband_simclr.py` | **CLI entry point** for multi-band SimCLR (5-band + multi-head NT-Xent + fine-tune) |
| `prepare_folds.py` | **Upstream**: merge train+val, generate stratified 5-fold CV splits |
| `trainer.py` | `Trainer` class: training loop, early stopping, device support, prediction saving |
| `pretrainer.py` | `Pretrainer` class: SimCLR contrastive training loop with augmentation + balance loss |
| `pretrainer_multiband.py` | `MultiBandPretrainer` class: multi-band contrastive training loop |
| `utils.py` | Data loaders (`create_dataloaders`, `create_pretrain_loaders`, `create_multi_pretrain_loaders`), device resolution, log/summary helpers |
| `model/` | Package with model definitions (`__init__.py` exports `MODEL_DICT`) |
| `data/TEST_DATASET.py` | PyTorch Dataset classes: `TrainDataset`, `FoldDataset`, `TestDataset`, `UnlabeledDataset`, `MultiUnlabeledDataset` |
| `train.ipynb` | Original Jupyter pipeline (deprecated in favor of CLI scripts) |
| `RNN_Exercise.py` | Legacy RNN/LSTM definitions (superseded by `model/rnn.py`) |
| `Results/{tag}/` | Per-run output: predictions.txt, model.pt, config.json, metrics.json, summary.txt, run.log |
| `Pretrained/{tag}/` | Pre-trained encoder weights: encoder.pt, adapter.pt, config.json, pretrain_metrics.json |

## Quick Commands (run from repo root)

```bash
# === Supervised training (0_run_train.py) ===
python 0_run_train.py --dataset MDD --model EEGLSTM --epochs 30 --lr 5e-4 --device cpu
python 0_run_train.py --dataset MDD --model EEGLSTM --epochs 30 --fold -1

# === SimCLR pre-training + fine-tuning (1_run_pretrain.py) ===
# Phase 1: pretrain only on MDD (50 epochs)
python 1_run_pretrain.py --phase 1 --action pretrain --dataset MDD --epochs_pretrain 50 --device cpu
# Phase 1: pretrain + auto-finetune (end-to-end)
python 1_run_pretrain.py --phase 1 --action both --dataset MDD --epochs_pretrain 50 --epochs_finetune 30 --patience 10
# Phase 1 + MoE
python 1_run_pretrain.py --phase 1 --action both --dataset MDD --use_moe --moe_num_experts 4 --epochs_pretrain 50
# Phase 2: multi-dataset pretrain
python 1_run_pretrain.py --phase 2 --action pretrain --epochs_pretrain 50 --device cpu
# Phase 2: finetune on a specific dataset
python 1_run_pretrain.py --phase 2 --action finetune --dataset MDD --pretrained_encoder Pretrained/multi_phase2_xxx/encoder.pt --pretrained_adapter Pretrained/multi_phase2_xxx/adapter.pt

# === Benchmark (2_run_benchmark.py) ===
python 2_run_benchmark.py --mode baseline --model EEGLSTM --epochs 30 --device cpu
python 2_run_benchmark.py --mode finetune --encoder Pretrained/multi_phase2_xxx/encoder.pt --adapter Pretrained/multi_phase2_xxx/adapter.pt
python 2_run_benchmark.py --mode compare

# === Multi-band SimCLR (3_run_multiband_simclr.py) ===
python 3_run_multiband_simclr.py --action pretrain --dataset MDD --epochs_pretrain 50
python 3_run_multiband_simclr.py --action both --dataset MDD --epochs_pretrain 50
python 3_run_multiband_simclr.py --action both --dataset MDD --use_moe --epochs_pretrain 50

# === Mamba backbone (via 0_run_train.py) ===
python 0_run_train.py --dataset MDD --model EEGMamba --epochs 30 --device cpu

# === KAN classifier variants ===
python 0_run_train.py --dataset MDD --model EEGLSTM_KAN --epochs 30
python 0_run_train.py --dataset MDD --model EEGNet_KAN --epochs 30

# === Attention-enhanced EEGNet ===
python 0_run_train.py --dataset MDD --model EEGNet_SimAM --epochs 30
python 0_run_train.py --dataset MDD --model EEGNet_SE --epochs 30

# === BCIC2A-optimized models ===
python 0_run_train.py --dataset BCIC2A --model ATCNet --lr 1e-3 --epochs 30 --device cpu
python 0_run_train.py --dataset BCIC2A --model ShallowConvNet --lr 1e-3 --epochs 30 --device cpu

# View dataset info
python -c "import json; info=json.load(open('data/MDD/dataset_info.json')); print(info['dataset']['category_list'], len(info['dataset']['channels']))"
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
| `--device` | cpu | Device: cpu, cuda, cuda:0, auto |

## `1_run_pretrain.py` Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--phase` | **required** | 1 = single-dataset, 2 = multi-dataset |
| `--action` | pretrain | pretrain / finetune / both |
| `--dataset` | MDD | Target dataset |
| `--datasets` | all 5 | Comma-separated list for Phase 2 pretrain |
| `--lr` | 5e-4 | Pre-training learning rate |
| `--epochs_pretrain` | 50 | Pre-training epochs |
| `--epochs_finetune` | 30 | Fine-tuning epochs |
| `--batch_size` | 256 | Batch size (larger for SimCLR) |
| `--temperature` | 0.1 | NT-Xent temperature |
| `--use_all_data` | False | Include val+test in pre-training |
| `--encoder_lr` | 5e-5 | Encoder LR during fine-tuning |
| `--classifier_lr` | 5e-4 | Classifier LR during fine-tuning |
| `--patience` | 0 | Early stopping patience (0=disabled) |
| `--use_moe` | False | Enable MoE layer |
| `--moe_num_experts` | 4 | Number of MoE experts |
| `--moe_top_k` | 2 | Top-k experts per token |
| `--balance_weight` | 0.01 | Load balancing loss weight |
| `--device` | cpu | Device: cpu, cuda, cuda:0, auto |

Each run creates an isolated output directory:
```
Results/MDD_EEGLSTM_pretrained_20260427_120000/
├── summary.txt         # human-readable config + results
├── run.log             # complete terminal output
├── config.json         # run configuration
├── metrics.json        # final/best val accuracy + loss history
├── predictions.txt     # test set predictions (one label per line)
└── model.pt            # trained model state_dict
```

Pre-training saves to `Pretrained/{tag}/`:
```
Pretrained/MDD_phase1_20260427_120000/
├── summary.txt
├── run.log
├── config.json
├── pretrain_metrics.json
└── encoder.pt          # LSTM (+ MoE) weights
```

## Architecture Notes

- All models expect input shape `(B, C, T)` — batch × channels × time
- RNN models transpose internally to `(B, T, C)` and use final hidden state for classification
- Device selection via `--device` flag (cpu / cuda / cuda:N / auto); falls back to CPU if CUDA unavailable
- Each dataset has different channel counts and class counts — configured automatically via `dataset_info.json`
- `--fold` requires `prepare_folds.py` to have been run first for that dataset
- BCIC2A uses `dataset_info_fixed.json` (handled automatically by `utils.py`)
- SimCLR pre-training discards labels; uses data augmentation for contrastive pairs
- MoE layer sits between LSTM encoder and projection/classifier head; discarded if not needed
- ChannelAdapter in Phase 2 uses per-dataset 1×1 Conv1d to unify channel counts

## `3_run_multiband_simclr.py` Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--action` | both | pretrain / finetune / both |
| `--dataset` | MDD | Target dataset (all 5 supported) |
| `--lr` | 5e-4 | Pre-training learning rate |
| `--epochs_pretrain` | 50 | Pre-training epochs |
| `--epochs_finetune` | 30 | Fine-tuning epochs |
| `--batch_size` | 256 | Batch size (larger for SimCLR) |
| `--temperature` | 0.1 | NT-Xent temperature |
| `--use_all_data` | False | Include val+test in pre-training |
| `--encoder_lr` | 5e-5 | Encoder LR during fine-tuning |
| `--classifier_lr` | 5e-4 | Classifier LR during fine-tuning |
| `--patience` | 0 | Early stopping patience (0=disabled) |
| `--use_moe` | False | Enable MoE layer |
| `--moe_num_experts` | 4 | Number of MoE experts |
| `--moe_top_k` | 2 | Top-k experts per token |
| `--balance_weight` | 0.01 | Load balancing loss weight |
| `--device` | auto | Device: cpu, cuda, cuda:0, auto |

## Known Results

### MDD dataset

| Model | Val Accuracy |
|-------|-------------|
| SimpleMLP | 75.00% |
| EEGNet (lr=1e-3) | 79.69% |
| EEGGRU (lr=5e-4) | 80.16% |
| EEGLSTM (lr=5e-4) | 81.25% |

SimCLR Pre-training + Fine-tuning (Phase 1, MDD): 50 epochs pre-training (lr=5e-4, batch=256, temp=0.1), 30 epochs fine-tuning (encoder_lr=5e-5, classifier_lr=5e-4, batch=256).

| Method | Val Accuracy |
|--------|-------------|
| SimCLR (no MoE) | **87.50%** |
| SimCLR + MoE (4 experts, top-2) | **86.41%** |

### BCIC2A dataset

Best single-fold results (fold 1):

| Model | Epochs | LR | Val Acc | Params |
|-------|--------|----|---------|--------|
| EEGNet | 50 | 5e-4 | 56.02% | 3.1K |
| EEGNet_SE | 50 | 5e-4 | 57.41% | 3.4K |
| ATCNet (nw=3) | 30 | 5e-4 | 57.87% | 75.5K |
| ATCNet (nw=5) | 30 | 5e-4 | 61.57% | 113.7K |
| **ATCNet (nw=5)** | **30** | **1e-3** | **66.20%** | **113.7K** |

5-fold CV (30 epochs): EEGNet **53.70% ± 3.19%**, EEGNet_SE **50.46% ± 1.92%**.

All BCIC2A results are archived in `Results/BCIC2A_optimization/`.

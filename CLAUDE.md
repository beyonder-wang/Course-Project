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

## Available Models

All defined in `train.ipynb`:
- **SimpleLinear** — flattened input → single linear layer (baseline)
- **SimpleMLP** — flattened input → hidden layers with ReLU+Dropout
- **EEGNet** — CNN with temporal/spatial/separable convolutions (standard EEG baseline)
- **EEGGRU** (aliased from `ExerciseEEGSimpleRNN` in `RNN_Exercise.py`) — bidirectional SimpleRNN
- **ExerciseEEGLSTM** (from `RNN_Exercise.py`) — bidirectional LSTM

## Key Files

| File | Purpose |
|------|---------|
| `train.ipynb` | Main pipeline: data loading, model definition, training loop, test inference |
| `RNN_Exercise.py` | RNN/LSTM model definitions (originally a student exercise with blanks) |
| `data/TEST_DATASET.py` | PyTorch Dataset classes for HDF5 data |
| `train.py` | (empty) — intended for refactored training script |
| `predict.py` | (empty) — intended for refactored prediction script |
| `test_result/{DATA_NAME}.txt` | Output: one predicted label per line |
| `tmp_results.md` | Log of model accuracy comparisons |

## Training Pipeline (in `train.ipynb`)

1. Set `DATA_NAME` to select dataset (MDD, BCIC2A, CHINESE, SEED, SLEEP)
2. Update `CHANNELS`, `CLASSES` to match dataset's `dataset_info.json`
3. Select a model (uncomment the desired model line)
4. Configure hyperparameters: `LR`, `EPOCHS`, `BATCH_SIZE`
5. Run all cells → trains model, logs val accuracy, saves predictions to `test_result/{DATA_NAME}.txt`

## Quick Commands (run from repo root)

```bash
# Train and predict — open and run train.ipynb in Jupyter / VS Code
# No CLI entry point yet (train.py and predict.py are stubs)

# View dataset info
python -c "import json; info=json.load(open('data/MDD/dataset_info.json')); print(info['dataset']['category_list'], len(info['dataset']['channels']))"

# Check saved predictions
cat test_result/MDD.txt | head -20
```

## Architecture Notes

- All models expect input shape `(B, C, T)` — batch × channels × time
- RNN models transpose internally to `(B, T, C)` and use final hidden state for classification
- Training loop runs on CPU (no CUDA setup currently)
- Each dataset has different channel counts and class counts — must be configured per-dataset
- The `CHANNELS` and `CLASSES` variables in cell 13 must match the target dataset

## Known Results (on MDD dataset)

| Model | Val Accuracy |
|-------|-------------|
| SimpleMLP | 75.00% |
| EEGNet (lr=1e-3) | 79.69% |
| EEGGRU (lr=5e-4) | 80.16% |
| ExerciseEEGLSTM (lr=5e-4) | 81.25% |

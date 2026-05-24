# Checkpoint Evaluation Results (Full)

## Key Finding

Old checkpoints (root directory) were trained on **RAW (unnormalized)** data.
The newer  was trained with **per-channel normalization**.

## Results

| # | Checkpoint | Model | Acc (raw) | Acc (normed) | Best Acc | Mode |
|---|---|---|---|---|---|---|
| 1 | hybrid_best_model.pth | EEGNetHybrid | 0.9391 | 0.5000 | **0.9391** | raw |
| 2 | ensemble_model_seed_123.pth | EEGNet | 0.9344 | 0.5000 | **0.9344** | raw |
| 3 | ensemble_model_seed_42.pth | EEGNet | 0.9297 | 0.5000 | **0.9297** | raw |
| 4 | outputs\eegnet_seed42_best.pth | EEGNet | 0.5672 | 0.9297 | **0.9297** | normed |
| 5 | ensemble_model_seed_2024.pth | EEGNet | 0.9266 | 0.5000 | **0.9266** | raw |
| 6 | ultimate_best_model.pth | EEGNet | 0.9203 | 0.5000 | **0.9203** | raw |
| 7 | best_model_fold_1.pth | EEGNet | 0.9156 | 0.5000 | **0.9156** | raw |
| 8 | best_model_fold_2.pth | EEGNet | 0.9156 | 0.5000 | **0.9156** | raw |
| 9 | best_model_fold_3.pth | EEGNet | 0.9047 | 0.5016 | **0.9047** | raw |
| 10 | best_model.pth | EEGNetOld | 0.9031 | 0.5000 | **0.9031** | raw |
| 11 | best_model_fold_4.pth | EEGNet | 0.8969 | 0.5000 | **0.8969** | raw |
| 12 | best_model_fold_5.pth | EEGNet | 0.8828 | 0.5000 | **0.8828** | raw |

## Summary

- Total checkpoints: 12
- All loaded successfully
- Best single model: **hybrid_best_model.pth** (EEGNetHybrid) with val acc = 0.9391
- 11/12 models use raw input, 1/12 uses normalized input
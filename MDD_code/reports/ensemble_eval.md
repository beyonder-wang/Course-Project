# Ensemble Evaluation Results

Total strategies evaluated: 37

| # | Strategy | Acc | Bal Acc | F1 |
|---|---|---|---|---|
| 1 | threshold-tuned top-3 weighted (t=0.61) ⭐ | 0.9563 | 0.9563 | 0.9562 |
| 2 | top-2 average | 0.9484 | 0.9484 | 0.9483 |
| 3 | acc>=0.93 (2 models) | 0.9484 | 0.9484 | 0.9483 |
| 4 | best exhaustive combo | 0.9484 | 0.9484 | 0.9483 |
| 5 | top-4 average | 0.9469 | 0.9469 | 0.9468 |
| 6 | top-5 average | 0.9422 | 0.9422 | 0.9421 |
| 7 | weighted top-5 | 0.9422 | 0.9422 | 0.9421 |
| 8 | weighted(acc-0.5) top-5 | 0.9422 | 0.9422 | 0.9421 |
| 9 | seed+hybrid (5 models) | 0.9422 | 0.9422 | 0.9421 |
| 10 | seed+hybrid weighted (5 models) | 0.9422 | 0.9422 | 0.9421 |
| 11 | top-3 average | 0.9406 | 0.9406 | 0.9405 |
| 12 | top-8 average | 0.9406 | 0.9406 | 0.9405 |
| 13 | acc>=0.91 (8 models) | 0.9406 | 0.9406 | 0.9405 |
| 14 | weighted top-3 | 0.9406 | 0.9406 | 0.9405 |
| 15 | weighted top-8 | 0.9406 | 0.9406 | 0.9405 |
| 16 | weighted(acc-0.5) top-3 | 0.9406 | 0.9406 | 0.9405 |
| 17 | weighted(acc-0.5) top-8 | 0.9406 | 0.9406 | 0.9405 |
| 18 | top-6 average | 0.9391 | 0.9391 | 0.9390 |
| 19 | acc>=0.92 (6 models) | 0.9391 | 0.9391 | 0.9390 |
| 20 | single: hybrid_best_model.pth | 0.9391 | 0.9391 | 0.9389 |
| 21 | top-10 average | 0.9375 | 0.9375 | 0.9373 |
| 22 | acc>=0.90 (10 models) | 0.9375 | 0.9375 | 0.9373 |
| 23 | single: ensemble_model_seed_123.pth | 0.9344 | 0.9344 | 0.9342 |
| 24 | top-12 average | 0.9328 | 0.9328 | 0.9326 |
| 25 | weighted top-12 | 0.9328 | 0.9328 | 0.9326 |
| 26 | single: eegnet_seed42_best.pth | 0.9297 | 0.9297 | 0.9295 |
| 27 | single: ensemble_model_seed_42.pth | 0.9297 | 0.9297 | 0.9295 |
| 28 | single: ensemble_model_seed_2024.pth | 0.9266 | 0.9266 | 0.9263 |
| 29 | seed ensemble (3 models) | 0.9266 | 0.9266 | 0.9263 |
| 30 | single: ultimate_best_model.pth | 0.9203 | 0.9203 | 0.9201 |
| 31 | fold ensemble (5 models) | 0.9187 | 0.9187 | 0.9185 |
| 32 | single: best_model_fold_2.pth | 0.9156 | 0.9156 | 0.9153 |
| 33 | single: best_model_fold_1.pth | 0.9156 | 0.9156 | 0.9151 |
| 34 | single: best_model_fold_3.pth | 0.9047 | 0.9047 | 0.9046 |
| 35 | single: best_model.pth | 0.9031 | 0.9031 | 0.9030 |
| 36 | single: best_model_fold_4.pth | 0.8969 | 0.8969 | 0.8963 |
| 37 | single: best_model_fold_5.pth | 0.8828 | 0.8828 | 0.8824 |

## Best Strategy

- **Strategy**: threshold-tuned top-3 weighted (t=0.61)
- **Accuracy**: 0.9563
- **Balanced Accuracy**: 0.9563
- **Macro F1**: 0.9562
- **Confusion Matrix**: [[301, 19], [9, 311]]

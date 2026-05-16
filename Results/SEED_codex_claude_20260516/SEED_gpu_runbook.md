# SEED GPU Runbook

This file records the most promising GPU-ready schemes implemented in this session.

## Recommended Scheme 1: SEEDGraphormer

Use this as the new primary heavy baseline on the multi-4090 machine.

```bash
python 0_run_train.py \
  --dataset SEED \
  --model SEEDGraphormer \
  --device cuda:0 \
  --amp \
  --batch_size 128 \
  --grad_accum_steps 1 \
  --epochs 120 \
  --patience 25 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --scheduler plateau \
  --plateau_patience 10 \
  --graphormer_dim 128 \
  --graphormer_depth 6 \
  --graphormer_heads 8 \
  --graphormer_mlp_ratio 4 \
  --graphormer_dropout 0.3 \
  --graphormer_attn_dropout 0.1 \
  --graphormer_top_k 12 \
  --graphormer_dyn_alpha 0.2 \
  --output_root Results/SEED_gpu_runs
```

Suggested sweep:

- `lr`: `1e-3`, `5e-4`
- `graphormer_dim`: `128`, `192`
- `graphormer_depth`: `6`, `8`
- `graphormer_top_k`: `12`, `16`
- `graphormer_dyn_alpha`: `0.15`, `0.20`, `0.25`

## Recommended Scheme 2: SEEDGraphormer + EmotionDL

Use this after the plain `SEEDGraphormer` baseline.

```bash
python 0_run_train.py \
  --dataset SEED \
  --model SEEDGraphormer \
  --device cuda:0 \
  --amp \
  --batch_size 128 \
  --grad_accum_steps 1 \
  --epochs 120 \
  --patience 25 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --scheduler plateau \
  --plateau_patience 10 \
  --graphormer_dim 128 \
  --graphormer_depth 6 \
  --graphormer_heads 8 \
  --graphormer_mlp_ratio 4 \
  --graphormer_dropout 0.3 \
  --graphormer_attn_dropout 0.1 \
  --graphormer_top_k 12 \
  --graphormer_dyn_alpha 0.2 \
  --emotion_dl_alpha 0.2 \
  --emotion_aux_weight 0.5 \
  --emotion_hidden_dim 128 \
  --emotion_dropout 0.1 \
  --output_root Results/SEED_gpu_runs
```

Suggested sweep:

- `emotion_dl_alpha`: `0.1`, `0.2`, `0.3`
- `emotion_aux_weight`: `0.3`, `0.5`, `1.0`

## Fallback Scheme: RGNN + EmotionDL

If you want a graph-only reference with lower memory cost, keep the earlier RGNN path as a fallback:

```bash
python 0_run_train.py \
  --dataset SEED \
  --model RGNN \
  --device cuda:0 \
  --amp \
  --batch_size 256 \
  --grad_accum_steps 1 \
  --epochs 120 \
  --patience 25 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --scheduler plateau \
  --plateau_patience 10 \
  --rgnn_top_k 8 \
  --rgnn_dyn_alpha 0.15 \
  --emotion_dl_alpha 0.2 \
  --emotion_aux_weight 0.5 \
  --output_root Results/SEED_gpu_runs
```

## Practical Notes

- `0_run_train.py` now supports `--amp` and `--grad_accum_steps`, so you can safely push batch size up on 4090.
- Every supervised run now auto-generates `summary.txt` next to `metrics.json`, `config.json`, `model.pt`, and `predictions.txt`.
- If you want to use more than one GPU immediately, the quickest low-risk approach is to launch separate sweeps on different devices:

```bash
python 0_run_train.py ... --device cuda:0 ...
python 0_run_train.py ... --device cuda:1 ...
python 0_run_train.py ... --device cuda:2 ...
python 0_run_train.py ... --device cuda:3 ...
```

- This is preferable to adding rushed multi-GPU wrapping before the methods themselves are validated.
- Keep all new outputs under one root such as `Results/SEED_gpu_runs/` so cleanup is easy afterward.

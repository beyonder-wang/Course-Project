# SEED GPU Runbook

This file records the most promising GPU-ready schemes implemented in this session.

## Recommended Scheme 1: SEEDBandGraphNet

Use this as the new primary SEED-specific scheme on the multi-4090 machine.
It pushes one step closer to the strongest SEED literature pattern under the
current repo constraints: per-band graph encoders, explicit `DASM/RASM`
asymmetry, band-attention fusion, and sample-adaptive dynamic graphs.

```bash
python 0_run_train.py \
  --dataset SEED \
  --model SEEDBandGraphNet \
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
  --bandgraph_hidden_dim 48 \
  --bandgraph_graph_layers 2 \
  --bandgraph_band_hidden 96 \
  --bandgraph_asym_hidden 96 \
  --bandgraph_fusion_hidden 192 \
  --bandgraph_dropout 0.3 \
  --bandgraph_top_k 10 \
  --bandgraph_dyn_alpha 0.2 \
  --output_root Results/SEED_gpu_runs
```

Suggested sweep:

- `lr`: `1e-3`, `5e-4`
- `bandgraph_hidden_dim`: `48`, `64`
- `bandgraph_band_hidden`: `96`, `128`
- `bandgraph_asym_hidden`: `96`, `128`
- `bandgraph_top_k`: `8`, `10`, `12`
- `bandgraph_dyn_alpha`: `0.15`, `0.20`, `0.25`

## Recommended Scheme 2: SEEDBandGraphNet + EmotionDL

Use this after the plain `SEEDBandGraphNet` baseline.

```bash
python 0_run_train.py \
  --dataset SEED \
  --model SEEDBandGraphNet \
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
  --bandgraph_hidden_dim 48 \
  --bandgraph_graph_layers 2 \
  --bandgraph_band_hidden 96 \
  --bandgraph_asym_hidden 96 \
  --bandgraph_fusion_hidden 192 \
  --bandgraph_dropout 0.3 \
  --bandgraph_top_k 10 \
  --bandgraph_dyn_alpha 0.2 \
  --emotion_dl_alpha 0.2 \
  --emotion_aux_weight 0.5 \
  --emotion_hidden_dim 128 \
  --emotion_dropout 0.1 \
  --output_root Results/SEED_gpu_runs
```

Suggested sweep:

- `emotion_dl_alpha`: `0.1`, `0.2`, `0.3`
- `emotion_aux_weight`: `0.3`, `0.5`, `1.0`

## Fallback Scheme 1: SEEDAsymNet + EmotionDL

`SEEDAsymNet` is still a strong fallback, and it has now been upgraded from a
batch-static graph to a sample-adaptive dynamic graph:

```bash
python 0_run_train.py \
  --dataset SEED \
  --model SEEDAsymNet \
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
  --seedasym_hidden_dim 64 \
  --seedasym_graph_layers 2 \
  --seedasym_asym_hidden 256 \
  --seedasym_fusion_hidden 256 \
  --seedasym_dropout 0.3 \
  --seedasym_top_k 8 \
  --seedasym_dyn_alpha 0.15 \
  --emotion_dl_alpha 0.2 \
  --emotion_aux_weight 0.5 \
  --emotion_hidden_dim 128 \
  --emotion_dropout 0.1 \
  --output_root Results/SEED_gpu_runs
```

## Fallback Scheme 2: RGNN + EmotionDL

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

## Deprioritized But Still Available: SEEDGraphormer

The heavier raw-window `SEEDGraphormer` path is still implemented, but it is no
longer the first recommendation because the first two GPU runs stayed well below
the simpler graph baselines:

- plain `SEEDGraphormer`: best val ACC `39.56%`
- `SEEDGraphormer + EmotionDL`: best val ACC `43.78%`

## Practical Notes

- `0_run_train.py` now supports `--amp` and `--grad_accum_steps`, so you can safely push batch size up on 4090.
- Every supervised run now auto-generates `summary.txt` next to `metrics.json`, `config.json`, `model.pt`, and `predictions.txt`.
- `trainer.py` now uses the newer AMP API when available, so newer PyTorch builds should not spam the old CUDA AMP deprecation warnings.
- If a remote run finishes without `summary.txt`, you can recover it from the run directory:

```bash
python tools/generate_run_summary.py --run_dir Results/SEED_gpu_runs/YOUR_RUN_DIR
```

- If you only have a raw log file, you can still produce a standalone summary:

```bash
python tools/generate_run_summary.py --log_path /path/to/run.log --summary_out /path/to/run_summary.txt
```
- If you want to use more than one GPU immediately, the quickest low-risk approach is to launch separate sweeps on different devices:

```bash
python 0_run_train.py ... --device cuda:0 ...
python 0_run_train.py ... --device cuda:1 ...
python 0_run_train.py ... --device cuda:2 ...
python 0_run_train.py ... --device cuda:3 ...
```

- This is preferable to adding rushed multi-GPU wrapping before the methods themselves are validated.
- Keep all new outputs under one root such as `Results/SEED_gpu_runs/` so cleanup is easy afterward.

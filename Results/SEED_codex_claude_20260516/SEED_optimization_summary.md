# SEED Optimization Summary

Generated: 2026-05-16

## Goal

- Reuse the Claude Code collaboration flow on `SEED`
- First compare literature guidance against the current repo
- Then optimize a SEED model under a CPU-only budget
- Stop if validation ACC exceeds `85%`, or otherwise leave the repo in a better state with reproducible artifacts

## Literature vs Repo

- The report's main recommendation is `DE/LDS spectral features -> graph modeling -> cross-subject alignment`.
- The repo originally focused on raw-waveform CNN/RNN models and did not include a SEED-oriented DE feature path or graph model.
- `sub_1.h5` has now been verified to contain `session_id`, `trial_id`, and per-segment `label/start_time/end_time` metadata on the `eeg` dataset itself.
- Because of that, the safest path for this round was:
  1. keep using the verified `train.h5/val.h5/test_x_only.h5` split
  2. add SEED-oriented models inside the current training pipeline
  3. benchmark them on CPU

## Code Added

- `DENet`
  - File: `model/simple.py`
  - Raw EEG -> 5-band FFT decomposition -> DE-style log-variance features -> MLP
- `DGCNN`
  - File: `model/dgcnn.py`
  - Raw EEG -> internal DE features -> learnable graph over channels -> graph classifier
  - Updated to support `EmotionDL` features on top of the strongest current baseline
  - Updated `DGCNN_RG` to use sample-adaptive dynamic graphs instead of a batch-averaged graph
- `RGNN`
  - File: `model/rgnn.py`
  - Raw EEG -> internal DE features -> biologically inspired sparse graph prior + dynamic DE-similarity adjacency
- `EmotionDL`
  - File: `model/emotion_dl.py`
  - Auxiliary label-distribution head for soft-target training on top of graph embeddings
- `SEEDGraphormer`
  - File: `model/seed_graphormer.py`
  - Heavy multiband graph-transformer using DE features, sparse graph priors and Transformer encoder layers
- `SEEDAsymNet`
  - File: `model/seed_asymnet.py`
  - Raw EEG -> internal `DE` features + `DASM/RASM` asymmetry + sparse prior graph fusion
  - Updated to use sample-adaptive dynamic graphs instead of one batch-shared graph
- `SEEDBandGraphNet`
  - File: `model/seed_bandgraph.py`
  - Raw EEG -> per-band graph encoders + per-band asymmetry modeling + band-attention fusion
- `0_run_train.py`
  - Added `--output_root` so all task artifacts can be grouped under one result subdirectory
  - Added supervised `--amp` and `--grad_accum_steps`
  - Added `EmotionDL` flags and `RGNN` graph-control flags
  - Added `SEEDGraphormer` flags
  - Added `SEEDAsymNet` flags
  - Added `SEEDBandGraphNet` flags
- `trainer.py`
  - Added support for model outputs carrying intermediate features
  - Added soft-target EmotionDL training path
  - Supervised runs now auto-write `summary.txt`
  - AMP path now prefers the newer `torch.amp` API when available
- `tools/generate_run_summary.py`
  - Recover `summary.txt` from a run directory
  - Or build a standalone summary directly from a raw `run.log`

## Experiments

All runs were saved under this task root:

- `Results/SEED_codex_claude_20260516/`

Summary of completed runs:

| Model | Key setup | Best Val ACC | Best Epoch | Notes |
| --- | --- | ---: | ---: | --- |
| `EEGNet_SE` | `lr=1e-3`, `epochs=5`, `standardize_inputs`, cosine | 38.44% | 3 | Raw-waveform baseline, failed to leave near-chance regime |
| `EEGLSTM` | `lr=5e-4`, `epochs=5`, plateau, mixup, label smoothing | 33.78% | 4 | Much slower on CPU and no useful lift |
| `DENet` | `lr=1e-3`, `epochs=10`, plateau | 41.56% | 1 | DE-style features alone were not enough on this split |
| `FBCNet` | `lr=1e-3`, `epochs=20`, plateau | 43.56% | 10 | Better than raw baselines, still well below target |
| `DGCNN` | `lr=1e-3`, `epochs=100`, plateau, `patience=20` | **45.78%** | 9 | Best result of this round; longer training did not close the gap |
| `DGCNN_RG` | `lr=1e-3`, `epochs=20`, plateau | 43.56% | 8 | RGNN-inspired dynamic residual adjacency + sparse graph + DropEdge; did not beat plain DGCNN |
| `RGNN` | `lr=1e-3`, `epochs=120`, plateau, GPU | 42.89% | 18 | User-run full GPU baseline; underperformed plain DGCNN on this split |
| `RGNN + EmotionDL` | `lr=1e-3`, `epochs=120`, plateau, GPU | 45.11% | 41 | Better than plain RGNN, but still below the best `DGCNN` |
| `SEEDGraphormer` | `lr=1e-3`, `epochs=120`, plateau, GPU | 39.56% | 4 | Heavy raw-window path, clearly not the best next investment |
| `SEEDGraphormer + EmotionDL` | `lr=1e-3`, `epochs=120`, plateau, GPU | 43.78% | 55 | EmotionDL helped, but not enough to justify this direction |
| `SEEDBandGraphNet` | `lr=1e-3`, `epochs=120`, plateau, GPU | 38.67% | 5 | New multi-band graph fusion idea, but the first real GPU runs underperformed badly |
| `SEEDBandGraphNet` | `lr=1e-3`, `epochs=120`, plateau, GPU | 38.67% | 7 | Repeat run confirmed the direction is weak on the current split |

Best preserved result directory:

- `Results/SEED_codex_claude_20260516/SEED_DGCNN_20260516_024431/`

## Conclusion

- The `85%` stop target was **not reached**.
- The bottleneck does **not** look like "insufficient epochs" or a small LR issue.
- Even after moving closer to the literature with internal DE features and a lightweight graph model, the current repo split remained far below paper-level SEED results.

Most likely reasons:

- the repo split is not aligned with the exact SEED evaluation settings used in the high-score papers
- the paper-grade pipelines rely on stronger ingredients than this repo currently has, especially:
  - official or better-matched `DE + LDS` features
  - more faithful `RGNN/SOGNN`-style graph inductive bias
  - cross-subject alignment or label-distribution regularization

Additional takeaway from the newest iteration:

- A lightweight RGNN-inspired adjacency refinement (`DGCNN_RG`) was implementable and CPU-cheap, but on this split it still failed to surpass the simpler `DGCNN`.
- That makes it less likely that the current gap is caused by "not enough graph flexibility" alone.

## Recommended Next Step

If continuing on a GPU machine, the most promising next move is:

1. run `DGCNN + EmotionDL` at full scale on GPU
2. run the upgraded sample-adaptive `DGCNN_RG + EmotionDL`
3. keep `SEEDAsymNet + EmotionDL` as the nearest lower-risk fallback
4. use the now-verified `sub_1.h5` metadata as the future entry point for a more official `DE/LDS`-style pipeline

## GPU-Ready Schemes Implemented

The two recommended schemes now implemented in the repo are:

1. `DGCNN + EmotionDL`
   - the strongest current baseline in this repo, now upgraded to support soft-target label-distribution training
2. `DGCNN_RG + EmotionDL`
   - the dynamic-graph DGCNN path, upgraded from a batch-shared graph to sample-adaptive dynamic graphs

## Smoke Verification

Minimal 1-epoch CPU smoke tests completed successfully:

- Plain RGNN:
  - `Results/SEED_codex_claude_20260516/smoke_plain/SEED_RGNN_20260516_143911/`
- RGNN + EmotionDL:
  - `Results/SEED_codex_claude_20260516/smoke_emotion/SEED_RGNN_20260516_143930/`
- Plain SEEDGraphormer:
  - `Results/SEED_codex_claude_20260516/smoke_graphormer/SEED_SEEDGraphormer_20260516_160809/`
- SEEDGraphormer + EmotionDL:
  - `Results/SEED_codex_claude_20260516/smoke_graphormer_emotion/SEED_SEEDGraphormer_20260516_160953/`
- Plain SEEDAsymNet:
  - `Results/SEED_codex_claude_20260516/smoke_asym/SEED_SEEDAsymNet_20260516_162505/`
- SEEDAsymNet + EmotionDL:
  - `Results/SEED_codex_claude_20260516/smoke_asym_emodl/SEED_SEEDAsymNet_20260516_162522/`
- Updated sample-adaptive SEEDAsymNet + EmotionDL:
  - `Results/SEED_codex_claude_20260516/smoke_asym_samplegraph/SEED_SEEDAsymNet_20260516_163824/`
- Plain SEEDBandGraphNet:
  - `Results/SEED_codex_claude_20260516/smoke_bandgraph/SEED_SEEDBandGraphNet_20260516_163751/`
- SEEDBandGraphNet + EmotionDL:
  - `Results/SEED_codex_claude_20260516/smoke_bandgraph_emodl/SEED_SEEDBandGraphNet_20260516_163950/`
- DGCNN + EmotionDL:
  - `Results/SEED_codex_claude_20260516/smoke_dgcnn_emodl/SEED_DGCNN_20260516_165205/`
- DGCNN_RG + EmotionDL:
  - `Results/SEED_codex_claude_20260516/smoke_dgcnnrg_emodl/SEED_DGCNN_RG_20260516_165205/`

## New Data Finding

- `data/SEED/SEED/sub_1.h5` is not just unlabeled raw storage.
- Each trial group carries `session_id` and `trial_id`.
- Each segment's `eeg` dataset carries `label`, `segment_id`, `start_time`, `end_time`, and `time_length`.
- That means the repo now has a verified path toward future session-aware feature caching or `DE/LDS`-style preprocessing, even though only `sub_1.h5` is currently present locally.

## Disk Usage

- This task result directory is far below the `15 GB` cleanup threshold
- One redundant short-run `DGCNN` directory was removed after being superseded by the longer run

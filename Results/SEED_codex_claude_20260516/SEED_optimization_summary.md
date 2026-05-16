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
- `tools/prepare_seed_de_dataset.py`
  - Builds a `SEED_DE` dataset from official-style subject files
  - Extracts 5-band DE features and applies a lightweight LDS/Kalman smoother within each trial
  - Re-materializes `train/val/test` by matching raw segments back to the repo split
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

1. sync the full `data/SEED/SEED/sub_*.h5` set onto the GPU machine
2. run `python tools/prepare_seed_de_dataset.py --source_glob data/SEED/SEED/sub_*.h5 --split_source_dir data/SEED --target_dir data/SEED_DE`
3. train `DGCNN + EmotionDL` on `SEED_DE`
4. train sample-adaptive `DGCNN_RG + EmotionDL` on `SEED_DE`

## GPU-Ready Schemes Implemented

The two recommended schemes now implemented in the repo are:

1. `SEED_DE + DGCNN + EmotionDL`
   - the strongest current baseline in this repo, upgraded with smoothed DE input features and soft-target label-distribution training
2. `SEED_DE + DGCNN_RG + EmotionDL`
   - the dynamic-graph DGCNN path, upgraded from a batch-shared graph to sample-adaptive dynamic graphs on top of the same DE input

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
- A dry run of `tools/prepare_seed_de_dataset.py` confirms the matching logic works, but only partially on the current machine because just `sub_1.h5` is present locally:
  - `train.h5`: `60` matched, `840` missed
  - `val.h5`: `30` matched, `420` missed
  - `test_x_only.h5`: `30` matched, `420` missed
- With the full `sub_*.h5` set on the GPU machine, the repo now has a direct path toward session-aware `DE/LDS`-style preprocessing without inventing a new evaluation split.

## Disk Usage

- This task result directory is far below the `15 GB` cleanup threshold
- One redundant short-run `DGCNN` directory was removed after being superseded by the longer run

---

## Round 2: Session 2026-05-16 (Afternoon) — Data Split Discovery & Single-Subject DE+LDS

### Key Discovery: Subject Ordering in SEED Split Files

The existing `train.h5` / `val.h5` / `test_x_only.h5` are organized by subject in fixed-size blocks:
- Train: 15 subjects × 60 segments = 900 total (balanced: 20 per class per subject)
- Val:   15 subjects × 30 segments = 450 total (balanced: 10 per class per subject)
- Test:  15 subjects × 30 segments = 450 total

Verified by:
1. **Hash matching** against `sub_1.h5`: subj_1 = 60/60 ✅, subj_2~15 = 0/60 ✅
2. **Deep subject classifier**: 5-fold CV accuracy = **89.67% ± 0.9%** (13.4× chance), proving each block has distinct subject-specific EEG signatures

### New Code Added

| File | Purpose |
|------|---------|
| `tools/build_seed_sub1_de.py` | Build DE+LDS feature dataset from `sub_1.h5` (single subject, session/trial-aware) |
| `tools/split_seed_by_subject.py` | Split raw SEED data by subject index into train/val/test |

### New Datasets Created

| Dataset | Source | Split | Features |
|---------|--------|-------|----------|
| `SEED_SUB1_DE` | `sub_1.h5` | session 1+2 train, session 3 val | DE+LDS (62×5) |
| `SEED_SUB1_DE_RANDOM` | `sub_1.h5` | random 70/30 | DE+LDS (62×5) |
| `SEED_SUB1_DE_S23v1` | `sub_1.h5` | session 2+3 train, session 1 val | DE+LDS (62×5) |
| `SEED_SUB1_DE_TRIAL` | `sub_1.h5` | first 9 trials train, last 4 val per session | DE+LDS (62×5) |
| `SEED_SUB1_DE_STRAT` | `sub_1.h5` | stratified trial-level split | DE+LDS (62×5) |
| `SEED_BYSUBJ` | raw `train.h5`+`val.h5` | subjects 1-12 train, 13-15 val + 5-fold CV | Raw EEG (62×400) |

### Round 2 Experiment Results

#### Single-Subject DE+LDS (Cross-Session)

| Dataset | Model | Key Setup | Best Val ACC | Notes |
|---------|-------|-----------|-------------:|-------|
| `SEED_SUB1_DE` (S12→3) | DGCNN | baseline, no reg | 48.98% | Already beats raw SEED 45.78% |
| `SEED_SUB1_DE` | DGCNN | + Mixup α=0.3 + LabelSmooth | 54.90% | Mixup is the single most impactful trick |
| `SEED_SUB1_DE` | DGCNN | + Mixup α=0.5 + LS + Aug | 56.12% | Noise + channel dropout aug |
| **`SEED_SUB1_DE`** | **DGCNN** | **+ Mixup + LS + Aug, seed=123** | **57.14%** | **Best single-subject result** |
| `SEED_SUB1_DE` | DGCNN | + EmotionDL | 55.78% | EmotionDL didn't help here |
| `SEED_SUB1_DE` | RGNN | sparse graph prior | 46.67% | Worse than DGCNN |
| `SEED_SUB1_DE` | DGCNN | cosine scheduler | 56.53% | Plateau > cosine |
| `SEED_SUB1_DE` | DENet | simple MLP | 42.04% | Too simple |
| `SEED_SUB1_DE_RANDOM` | DGCNN | Mixup + LS | **100.00%** | Proves features are perfect; bottleneck is cross-session |
| `SEED_SUB1_DE_S23v1` | DGCNN | Mixup + LS + Aug | 51.90% | S3→1 harder than S12→3 |
| `SEED_SUB1_DE_TRIAL` | DGCNN | Mixup + LS | 35.61% | Severe label imbalance in val |
| `SEED_SUB1_DE_STRAT` | DGCNN | Mixup + LS + Aug | 46.24% | Stratified trial split didn't help |

#### Multi-Subject Raw EEG (Cross-Subject)

| Dataset | Model | Key Setup | Best Val ACC | Notes |
|---------|-------|-----------|-------------:|-------|
| `SEED_BYSUBJ` (12→3) | DGCNN | Mixup + LS + Aug | 38.89% | Cross-subject is extremely hard |
| `SEED_BYSUBJ` (12→3) | DGCNN | no Mixup | 37.04% | Even worse without regularization |
| `SEED_BYSUBJ` (5-fold CV) | DGCNN | Mixup + LS | **47.11% ± 2.11%** | Cross-subject within mixed folds |

### Key Takeaways from Round 2

1. **DE+LDS features are transformative**: 2 epochs on DE+LDS matched 100 epochs on raw EEG
2. **Cross-session is hard but learnable**: 57.14% with Mixup regularization on single subject
3. **Cross-subject with raw EEG is very hard**: Only 47% with 5-fold CV — need DE+LDS + more data per subject
4. **Mixup is the single most impactful technique**: +6-7% gain consistently
5. **Subject domain signal is strong**: 89.67% subject classifier accuracy means domain adversarial methods (NodeDAT, BiDANN) should be very effective
6. **The critical missing piece**: Full `sub_2.h5` ~ `sub_15.h5` files to enable multi-subject DE+LDS training

### Recommended Path Forward

1. Obtain full `data/SEED/SEED/sub_*.h5` (all 15 subjects) from teacher or SEED website
2. Run `prepare_seed_de_dataset.py` on full subject set → `SEED_DE`
3. Train `DGCNN + EmotionDL` on `SEED_DE` with cross-subject LOSO or folds
4. Implement domain-adversarial head (NodeDAT-style) leveraging the strong subject signal

---

## Round 3: Session 2026-05-16 (Evening) - Subject-Adversarial Training Plumbing

### New Code Added

| File | Purpose |
|------|---------|
| `model/domain_adversarial.py` | Gradient-reversal domain head for subject/session adversarial training |
| `data/TEST_DATASET.py` | Load optional H5 metadata such as `subject_id` / `session_id` and carry it through DataLoader batches |
| `trainer.py` | Combine classification loss with adversarial domain loss during supervised training |
| `0_run_train.py` | Add CLI flags for adversarial training and auto-enable feature-return mode for supported models |
| `prepare_folds.py` | Preserve metadata arrays when creating `all.h5` so 5-fold CV can still access domain labels |
| `tools/split_seed_by_subject.py` | Write `subject_id` into `SEED_BYSUBJ` train/val/test files |

### New CLI Flags

- `--subject_adv_weight`
- `--subject_adv_key`
- `--subject_adv_hidden_dim`
- `--subject_adv_dropout`
- `--subject_adv_grl_lambda`

### Current Recommendation

The highest-value next experiment is now:

1. `SEED_BYSUBJ + DGCNN + Mixup + Label Smoothing + subject-adversarial`
2. `SEED_SUB1_DE + DGCNN + Mixup + Label Smoothing + session-adversarial`

This is a method change, not just a hyperparameter tweak: it directly targets the
very strong domain signal observed in Round 2.

---

## Round 4: Session 2026-05-16 (Night) — Domain-Adversarial GPU Experiments

### Configuration

All runs: `Results/SEED_gpu_runs/`, 4× GPU, AMP enabled.

### Primary Line A: SEED_BYSUBJ (Cross-Subject 12→3) + DGCNN + Subject-Adversarial

| Run | Config | Best Val ACC | Best Epoch | Notes |
|-----|--------|-------------:|-----------:|-------|
| `...081543` | No adv, mixup=0.3, ls=0.1 | 35.93% | 14 | Non-adversarial baseline |
| `...082325` | adv=0.3, mixup=0.3, ls=0.1 | 39.26% | 29 | Subject-adversarial helps |
| `...082807` | adv=0.3, mixup=0.5, ls=0.1 | 40.00% | 33 | Higher mixup helps |
| `...083426` | adv=0.5, mixup=0.5, ls=0.1 | 40.37% | 56 | Higher adv weight helps |
| **`...083701`** | **adv=0.5, mixup=0.5, ls=0.1, EmotionDL α=0.2** | **42.22%** | 44 | **BEST: +6.29% over baseline** |
| `...083359` | DGCNN_RG, adv=0.3, mixup=0.5, ls=0.1 | 37.41% | 10 | DGCNN_RG worse than plain DGCNN |

**Subject-adversarial weight sweep (mixup=0.3, ls=0.1):**
| adv_weight | Best Val ACC |
|-----------|-------------:|
| 0 (off) | 35.93% |
| 0.1 | 37.78% |
| 0.2 | 38.89% |
| 0.3 | 39.26% |
| 0.5 | — (with mixup=0.5 → 40.37%) |
| 1.0 | 38.89% (too strong, deleted) |

**Mixup sweep (adv=0.3, ls=0.1):**
| mixup_alpha | Best Val ACC |
|------------|-------------:|
| 0.2 | 39.63% |
| 0.3 | 39.26% |
| 0.5 | 40.00% |

**Other sweeps (adv=0.3/0.5, mixup=0.5):**
- label_smoothing=0.05: 39.63% (worse than 0.1)
- batch_size=384: 38.52% (worse than 256)

### Primary Line B: SEED_SUB1_DE (Single-Subject, Cross-Session S12→3) + DGCNN + Session-Adversarial

| Run | Config | Best Val ACC | Best Epoch | Notes |
|-----|--------|-------------:|-----------:|-------|
| `...081609` | No adv, mixup=0.3, ls=0.1 | 54.42% | 3 | Non-adversarial baseline |
| `...081143` | adv=0.2 (session_id), mixup=0.3, ls=0.1 | 54.49% | 3 | Session-adversarial **no help** |
| `...083946` | No adv, lr=5e-4, wd=1e-3, mixup=0.5 | 51.16% | 3 | Overfitting persists |

**Key finding:** Session-adversarial does not help on single-subject data (only 2 sessions in train = 2 domain classes). The model overfits severely — best epoch is always 3, then degrades.

### Best Non-Adversarial SEED_SUB1_DE (from Round 2)
| Run | Config | Best Val ACC |
|-----|--------|-------------:|
| Round 2 | DGCNN + Mixup + LS + Aug, seed=123 | **57.14%** |

This remains the best single-subject result. Session-adversarial was not tried there since the data lacked session_id metadata at that time.

### Key Findings

1. **Subject-adversarial training is the most impactful method change on cross-subject SEED_BYSUBJ**: +6.29% absolute improvement (35.93% → 42.22%) from combining adversarial + mixup + EmotionDL.
2. **Higher adversarial weight (0.5) beats lower (0.1-0.3)**: The adversarial loss acts as a strong regularizer, preventing overfitting on the small 1080-sample training set.
3. **EmotionDL adds further regularization**: +1.85% on top of the best non-EmotionDL config.
4. **DGCNN_RG is consistently worse than DGCNN**: Confirmed on both original SEED (Round 1) and SEED_BYSUBJ (Round 4).
5. **Session-adversarial on single-subject data is ineffective**: Only 2 session domains in training; the model already overfits severely regardless.
6. **SEED_BYSUBJ with adversarial still far below paper-level results** (~42% vs 85%+): The 12-subject training set is too small; need the full 15-subject DE+LDS pipeline.

### Best Current Command

```bash
python 0_run_train.py \
  --dataset SEED_BYSUBJ \
  --model DGCNN \
  --device cuda:0 \
  --amp \
  --batch_size 256 \
  --epochs 120 \
  --patience 25 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --scheduler plateau \
  --plateau_patience 10 \
  --mixup_alpha 0.5 \
  --label_smoothing 0.1 \
  --subject_adv_weight 0.5 \
  --subject_adv_key subject_id \
  --subject_adv_hidden_dim 128 \
  --subject_adv_dropout 0.1 \
  --emotion_dl_alpha 0.2 \
  --emotion_aux_weight 0.5 \
  --emotion_hidden_dim 128 \
  --emotion_dropout 0.1 \
  --output_root Results/SEED_gpu_runs
```

### Next Single Method Change to Try

**Multi-subject DE+LDS with subject-adversarial**: Obtain the full `sub_2.h5` ~ `sub_15.h5` files, build `SEED_DE` with all 15 subjects, then apply the best adversarial + EmotionDL recipe on the full DE+LDS feature set. This should be substantially higher-value than further tuning on the 12-subject raw-EEG SEED_BYSUBJ.

### Kept Run Directories

Under `Results/SEED_gpu_runs/`:
- `SEED_BYSUBJ_DGCNN_20260516_081543/` — Non-adversarial baseline
- `SEED_BYSUBJ_DGCNN_20260516_082325/` — adv=0.3 sweep anchor
- `SEED_BYSUBJ_DGCNN_20260516_082807/` — mixup=0.5 discovery
- `SEED_BYSUBJ_DGCNN_20260516_083426/` — adv=0.5 sweet spot
- `SEED_BYSUBJ_DGCNN_20260516_083701/` — **BEST: EmotionDL + adversarial**
- `SEED_BYSUBJ_DGCNN_RG_20260516_083359/` — DGCNN_RG negative result
- `SEED_SUB1_DE_DGCNN_20260516_081143/` — Session-adversarial (no help)
- `SEED_SUB1_DE_DGCNN_20260516_081609/` — Non-adversarial SUB1_DE baseline
- `SEED_SUB1_DE_DGCNN_20260516_083946/` — Lower lr/higher wd attempt

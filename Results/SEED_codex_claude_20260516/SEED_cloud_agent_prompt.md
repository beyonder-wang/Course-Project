# SEED Cloud Agent Prompt

Use this prompt with the cloud repo agent after pulling the latest `main_jxWang`.

```text
You are working inside the Course-Project repo on branch main_jxWang.

Goal:
- Improve SEED results with priority on method changes over pure hyperparameter tuning.
- Use the newly implemented subject/session adversarial training path instead of introducing another large raw-window backbone.

Constraints:
- Preserve all prediction txt files and any run directory that is best, runner-up, or methodologically distinct.
- Save all new experiments under Results/SEED_gpu_runs/.
- Generate or keep summary.txt for every finished run.
- Stop a batch once the next higher-value method step is clear; do not wander into unrelated datasets.

Read first:
- Results/SEED_codex_claude_20260516/SEED_optimization_summary.md
- Results/SEED_codex_claude_20260516/SEED_gpu_runbook.md
- 0_run_train.py
- trainer.py
- data/TEST_DATASET.py
- model/domain_adversarial.py

Verify before training:
1. If using SEED_BYSUBJ, confirm train.h5 and val.h5 contain subject_id.
2. If using --fold on SEED_BYSUBJ, regenerate folds with the updated prepare_folds.py and confirm all.h5 also contains subject_id.
3. If using SEED_SUB1_DE, confirm train.h5 and val.h5 contain session_id.

Primary experiments:
1. SEED_BYSUBJ + DGCNN + mixup + label smoothing + subject-adversarial
2. SEED_SUB1_DE + DGCNN + mixup + label smoothing + session-adversarial

Start with:
python 0_run_train.py --dataset SEED_BYSUBJ --model DGCNN --device cuda:0 --amp --batch_size 256 --epochs 120 --patience 25 --lr 1e-3 --weight_decay 1e-4 --scheduler plateau --plateau_patience 10 --mixup_alpha 0.3 --label_smoothing 0.1 --subject_adv_weight 0.2 --subject_adv_key subject_id --subject_adv_hidden_dim 128 --subject_adv_dropout 0.1 --output_root Results/SEED_gpu_runs

python 0_run_train.py --dataset SEED_SUB1_DE --model DGCNN --device cuda:0 --amp --batch_size 256 --epochs 120 --patience 25 --lr 1e-3 --weight_decay 1e-4 --scheduler plateau --plateau_patience 10 --mixup_alpha 0.3 --label_smoothing 0.1 --subject_adv_weight 0.2 --subject_adv_key session_id --subject_adv_hidden_dim 128 --subject_adv_dropout 0.1 --output_root Results/SEED_gpu_runs

Focused sweep order:
- subject_adv_weight: 0.1, 0.2, 0.3
- mixup_alpha: 0.2, 0.3, 0.5
- label_smoothing: 0.05, 0.1
- batch_size: 256, 384

Only after the two primary lines are understood:
- try DGCNN_RG with the same adversarial setup
- keep EmotionDL as a lower-priority additive ablation, not the first branch

After each run:
- summarize best val acc, epoch, and whether subject/session adversarial helped vs the closest non-adversarial baseline
- update Results/SEED_codex_claude_20260516/SEED_optimization_summary.md with only high-signal findings
- delete only clearly redundant low-value runs created in this batch

Return format:
1. Ranked findings
2. Best current command
3. Next single method change to try
4. Paths to kept result directories
```

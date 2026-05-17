"""Multi-config GPU sweep for BCIC2A.

Launches training runs for a set of model configurations (architectures, presets,
seeds) and reports results. Designed to run on a GPU (RTX 4090) with AMP.

Usage:
    # Full sweep (sequential)
    python run_bcic2a_gpu_sweep.py

    # Single configuration test (1 epoch smoke-test)
    python run_bcic2a_gpu_sweep.py --smoke_test
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

import torch

from model import MODEL_DICT


PYTHON = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "Results")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "BCIC2A_gpu_sweep_summary.md")

# ---------------------------------------------------------------------------
# Configuration presets
# ---------------------------------------------------------------------------

# Known-good baseline: exact recipe that gave 68.06% on CPU
# Only minimal GPU-appropriate changes: device cuda, amp, more epochs
BASELINE_ARGS = [
    ("--dataset", "BCIC2A"),
    ("--model", "ATCNet"),
    ("--fold", "1"),
    ("--epochs", "300"),
    ("--batch_size", "32"),
    ("--lr", "1e-3"),
    ("--device", "cuda"),
    ("--amp",),
    ("--mixup_alpha", "0.2"),
    ("--scheduler", "plateau"),
    ("--plateau_patience", "15"),
    ("--plateau_factor", "0.9"),
    ("--plateau_min_lr", "1e-4"),
    ("--patience", "35"),
]

# Incremental improvements — add one thing at a time
IMPROVE_LARGER = BASELINE_ARGS + [
    ("--atc_preset", "large"),
    ("--lr", "1e-3"),
]

IMPROVE_WD = BASELINE_ARGS + [
    ("--weight_decay", "1e-4"),
]

IMPROVE_AUG = BASELINE_ARGS + [
    ("--aug_noise_std", "0.05"),
    ("--label_smoothing", "0.05"),
]

IMPROVE_CLIP_WARMUP = BASELINE_ARGS + [
    ("--grad_clip_norm", "1.0"),
    ("--warmup_epochs", "10"),
]

IMPROVE_FULL = BASELINE_ARGS + [
    ("--atc_preset", "large"),
    ("--weight_decay", "1e-4"),
    ("--aug_noise_std", "0.05"),
    ("--label_smoothing", "0.05"),
    ("--grad_clip_norm", "1.0"),
    ("--warmup_epochs", "10"),
    ("--batch_size", "64"),
]

CONFORMER_ARGS = [
    ("--dataset", "BCIC2A"),
    ("--model", "EEGConformer"),
    ("--fold", "1"),
    ("--epochs", "300"),
    ("--batch_size", "64"),
    ("--lr", "5e-4"),
    ("--device", "cuda"),
    ("--amp",),
    ("--weight_decay", "1e-4"),
    ("--grad_clip_norm", "1.0"),
    ("--warmup_epochs", "15"),
    ("--mixup_alpha", "0.2"),
    ("--label_smoothing", "0.05"),
    ("--scheduler", "cosine"),
    ("--patience", "80"),
    ("--conf_dim", "64"),
    ("--conf_blocks", "4"),
    ("--conf_heads", "4"),
    ("--conf_kernel", "31"),
    ("--conf_ff_expansion", "4"),
    ("--conf_patch_kernel", "25"),
    ("--conf_patch_stride", "10"),
    ("--conf_dropout", "0.1"),
]

SWEEP_CONFIGS = [
    # (name, args_list, seeds)
    ("01-baseline-original", BASELINE_ARGS, [21, 37]),
    ("02-baseline-large", IMPROVE_LARGER, [37]),
    ("03-weight-decay", IMPROVE_WD, [21, 37]),
    ("04-aug-ls", IMPROVE_AUG, [21, 37]),
    ("05-clip-warmup", IMPROVE_CLIP_WARMUP, [21, 37]),
    ("06-full-combo", IMPROVE_FULL, [21, 37]),
    ("07-eegconformer", CONFORMER_ARGS, [29, 43]),
]

STOP_TARGET = 0.75  # stop sweep early if any run exceeds this


def _flatten_args(pairs, seed):
    cmd = [PYTHON, "0_run_train.py"]
    for pair in pairs:
        cmd.extend(pair)
    cmd.extend(["--seed", str(seed)])
    return cmd


def _latest_result_dir(before):
    after = {
        name for name in os.listdir(RESULTS_DIR)
        if name.startswith("BCIC2A_")
    }
    new_dirs = sorted(after - before)
    if not new_dirs:
        return None
    return os.path.join(RESULTS_DIR, new_dirs[-1])


def _write_summary(rows):
    if not rows:
        return
    best = max(rows, key=lambda x: x["best_val_accuracy"])
    lines = [
        "# BCIC2A GPU Sweep Results",
        "",
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Best single model:** `{best['best_val_accuracy'] * 100:.2f}%` "
        f"({best['config']}, seed {best['seed']})",
        "",
        "| Config | Seed | Best Val Acc | Best Epoch | Run Dir |",
        "|:---|---:|---:|---:|:---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['seed']} | "
            f"{row['best_val_accuracy'] * 100:.2f}% | {row['best_epoch']} | "
            f"`{os.path.basename(row['run_dir'])}` |"
        )

    # Best ensemble combination (top-3 by val acc)
    sorted_rows = sorted(rows, key=lambda x: x["best_val_accuracy"], reverse=True)
    top3 = sorted_rows[:3]
    if len(top3) >= 2:
        ensemble_acc = sum(r["best_val_accuracy"] for r in top3) / len(top3)
        # This is an approximation — the actual ensemble may differ
        lines.extend([
            "",
            f"**Top-3 average (ensemble estimate):** `{ensemble_acc * 100:.2f}%`",
            f"**Stop target ({STOP_TARGET * 100:.0f}%):** "
            f"{'REACHED' if best['best_val_accuracy'] >= STOP_TARGET else 'Not yet'}",
        ])

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary written to: {SUMMARY_PATH}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BCIC2A GPU sweep")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run 1 epoch per config to verify the pipeline")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for config_name, config_args, seeds in SWEEP_CONFIGS:
        for seed in seeds:
            cmd = _flatten_args(config_args, seed)
            if args.smoke_test:
                # Override epochs to 1 for smoke test
                cmd[cmd.index("--epochs") + 1] = "1"
                cmd[cmd.index("--patience") + 1] = "0"

            before = set(os.listdir(RESULTS_DIR))
            desc = f"{config_name} (seed {seed})"
            print(f"\n{'=' * 60}")
            print(f"  Running: {desc}")
            print(f"  Command: {' '.join(cmd)}")
            print(f"{'=' * 60}\n")

            t_start = time.time()
            proc = subprocess.run(cmd, cwd=ROOT)
            elapsed = time.time() - t_start

            if proc.returncode != 0:
                print(f"[WARN] {desc} failed (return code {proc.returncode})")
                time.sleep(1)
                continue

            run_dir = _latest_result_dir(before)
            if run_dir is None:
                print(f"[WARN] {desc}: could not find run directory")
                time.sleep(1)
                continue

            with open(os.path.join(run_dir, "metrics.json"), "r") as f:
                metrics = json.load(f)

            val_acc = metrics["best_val_accuracy"]
            print(f"  {desc}: {val_acc * 100:.2f}% (epoch {metrics['best_epoch']}) "
                  f"[{elapsed / 60:.1f} min]")

            rows.append({
                "config": config_name,
                "seed": seed,
                "run_dir": run_dir,
                "best_val_accuracy": val_acc,
                "best_epoch": metrics["best_epoch"],
            })
            _write_summary(rows)

            if not args.smoke_test and val_acc >= STOP_TARGET:
                print(f"\n*** Reached stop target {STOP_TARGET * 100:.0f}%! ***")
                _write_summary(rows)
                return

            time.sleep(2)  # cooldown between runs

    _write_summary(rows)
    print(f"\nSweep complete. {len(rows)} runs total.")


if __name__ == "__main__":
    main()

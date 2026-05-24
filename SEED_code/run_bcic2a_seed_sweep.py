"""Seed sweep for the best current BCIC2A ATCNet recipe."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime


PYTHON = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "Results")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "BCIC2A_seed_sweep_summary.md")

SEEDS = [21, 29, 37, 43, 97]
BASE_CMD = [
    PYTHON, "0_run_train.py",
    "--dataset", "BCIC2A",
    "--model", "ATCNet",
    "--fold", "1",
    "--epochs", "120",
    "--lr", "1e-3",
    "--device", "cpu",
    "--mixup_alpha", "0.2",
    "--scheduler", "plateau",
    "--plateau_patience", "15",
    "--plateau_factor", "0.9",
    "--plateau_min_lr", "1e-4",
    "--patience", "35",
]


def _latest_result_dir(before):
    after = {
        name for name in os.listdir(RESULTS_DIR)
        if name.startswith("BCIC2A_ATCNet_")
    }
    new_dirs = sorted(after - before)
    if not new_dirs:
        return None
    return os.path.join(RESULTS_DIR, new_dirs[-1])


def _write_summary(rows):
    best = max(rows, key=lambda x: x["best_val_accuracy"]) if rows else None
    lines = [
        "# BCIC2A Seed Sweep",
        "",
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    if best is not None:
        lines.extend([
            f"Current best: `{best['best_val_accuracy'] * 100:.2f}%` at seed `{best['seed']}`",
            "",
        ])
    lines.extend([
        "| Seed | Best Val Acc | Best Epoch | Run Dir |",
        "|---:|---:|---:|---|",
    ])
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['best_val_accuracy'] * 100:.2f}% | "
            f"{row['best_epoch']} | `{os.path.basename(row['run_dir'])}` |"
        )
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for seed in SEEDS:
        before = set(os.listdir(RESULTS_DIR))
        cmd = BASE_CMD + ["--seed", str(seed)]
        print("Running:", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            print(f"[WARN] seed {seed} failed", flush=True)
            time.sleep(1)
            continue

        run_dir = _latest_result_dir(before)
        if run_dir is None:
            print(f"[WARN] could not find run dir for seed {seed}", flush=True)
            time.sleep(1)
            continue

        with open(os.path.join(run_dir, "metrics.json"), "r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append({
            "seed": seed,
            "run_dir": run_dir,
            "best_val_accuracy": metrics["best_val_accuracy"],
            "best_epoch": metrics["best_epoch"],
        })
        _write_summary(rows)
        if metrics["best_val_accuracy"] >= 0.70:
            print("Reached 70% stop condition.", flush=True)
            break
        time.sleep(1)


if __name__ == "__main__":
    main()

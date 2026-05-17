"""BCIC2A Round 5: narrow sweep around the first 70% ATCNet run.

Round 4 showed that the "paper-style" regularization recipe is useful:
  - paper_wd009_seed37 -> 70.37%
  - paper_wd009_seed97 -> 68.98%
  - S&R augmentation hurt or was unstable

This round keeps the model recipe fixed and tests only:
  1. additional seeds for ensemble diversity / stability
  2. a narrow weight-decay neighborhood around 0.009 on the best seed
"""

import json
import os
import subprocess
import sys
from datetime import datetime


PYTHON = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "Results")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "BCIC2A_round5_summary.md")

BASE_CMD = [
    PYTHON, "0_run_train.py",
    "--dataset", "BCIC2A",
    "--model", "ATCNet",
    "--fold", "1",
    "--epochs", "180",
    "--batch_size", "32",
    "--lr", "1e-3",
    "--device", "auto",
    "--amp",
    "--mixup_alpha", "0.2",
    "--scheduler", "plateau",
    "--plateau_patience", "20",
    "--plateau_factor", "0.9",
    "--plateau_min_lr", "1e-4",
    "--patience", "60",
    "--atc_preset", "paper",
]

EXPERIMENTS = [
    {
        "name": "paper_wd009_seed21",
        "cmd": BASE_CMD + ["--weight_decay", "0.009", "--seed", "21"],
    },
    {
        "name": "paper_wd009_seed29",
        "cmd": BASE_CMD + ["--weight_decay", "0.009", "--seed", "29"],
    },
    {
        "name": "paper_wd009_seed42",
        "cmd": BASE_CMD + ["--weight_decay", "0.009", "--seed", "42"],
    },
    {
        "name": "paper_wd009_seed43",
        "cmd": BASE_CMD + ["--weight_decay", "0.009", "--seed", "43"],
    },
    {
        "name": "paper_wd006_seed37",
        "cmd": BASE_CMD + ["--weight_decay", "0.006", "--seed", "37"],
    },
    {
        "name": "paper_wd012_seed37",
        "cmd": BASE_CMD + ["--weight_decay", "0.012", "--seed", "37"],
    },
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
    lines = [
        "# BCIC2A Round 5 Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Experiment | Best Val Acc | Best Epoch | Run Dir |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['best_val_accuracy'] * 100:.2f}% | "
            f"{row['best_epoch']} | `{os.path.basename(row['run_dir'])}` |"
        )
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for exp in EXPERIMENTS:
        before = set(os.listdir(RESULTS_DIR))
        print(f"\n=== Running {exp['name']} ===")
        print(" ".join(exp["cmd"]))
        proc = subprocess.run(exp["cmd"], cwd=ROOT)
        if proc.returncode != 0:
            print(f"[WARN] Experiment failed: {exp['name']}")
            continue

        run_dir = _latest_result_dir(before)
        if run_dir is None:
            print(f"[WARN] Could not determine result directory for {exp['name']}")
            continue

        with open(os.path.join(run_dir, "metrics.json"), "r", encoding="utf-8") as f:
            metrics = json.load(f)

        rows.append({
            "name": exp["name"],
            "run_dir": run_dir,
            "best_val_accuracy": metrics["best_val_accuracy"],
            "best_epoch": metrics["best_epoch"],
        })
        _write_summary(rows)

        print(
            f"Best val acc for {exp['name']}: "
            f"{metrics['best_val_accuracy'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()

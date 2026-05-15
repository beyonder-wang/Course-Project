"""BCIC2A Round 3 optimizer: protocol-aware ATCNet experiments.

Runs a small sequence of BCIC2A experiments centered on train-split
standardization and stronger regularization/augmentation for ATCNet.
"""

import json
import os
import subprocess
import sys
from datetime import datetime


PYTHON = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "Results")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "BCIC2A_round3_summary.md")


EXPERIMENTS = [
    {
        "name": "atcnet_std_cosine_ls",
        "cmd": [
            PYTHON, "0_run_train.py",
            "--dataset", "BCIC2A",
            "--model", "ATCNet",
            "--fold", "1",
            "--epochs", "35",
            "--lr", "1e-3",
            "--device", "cpu",
            "--standardize_inputs",
            "--label_smoothing", "0.1",
            "--scheduler", "cosine",
            "--weight_decay", "1e-3",
            "--patience", "10",
        ],
    },
    {
        "name": "atcnet_std_cosine_ls_mixup",
        "cmd": [
            PYTHON, "0_run_train.py",
            "--dataset", "BCIC2A",
            "--model", "ATCNet",
            "--fold", "1",
            "--epochs", "35",
            "--lr", "1e-3",
            "--device", "cpu",
            "--standardize_inputs",
            "--label_smoothing", "0.1",
            "--scheduler", "cosine",
            "--weight_decay", "1e-3",
            "--mixup_alpha", "0.2",
            "--patience", "10",
        ],
    },
    {
        "name": "atcnet_std_cosine_ls_mixup_aug",
        "cmd": [
            PYTHON, "0_run_train.py",
            "--dataset", "BCIC2A",
            "--model", "ATCNet",
            "--fold", "1",
            "--epochs", "35",
            "--lr", "1e-3",
            "--device", "cpu",
            "--standardize_inputs",
            "--label_smoothing", "0.1",
            "--scheduler", "cosine",
            "--weight_decay", "1e-3",
            "--mixup_alpha", "0.2",
            "--aug_noise_std", "0.01",
            "--aug_time_shift", "16",
            "--patience", "10",
        ],
    },
    {
        "name": "atcnet7_std_cosine_ls_mixup",
        "cmd": [
            PYTHON, "0_run_train.py",
            "--dataset", "BCIC2A",
            "--model", "ATCNet",
            "--fold", "1",
            "--epochs", "35",
            "--lr", "1e-3",
            "--device", "cpu",
            "--standardize_inputs",
            "--label_smoothing", "0.1",
            "--scheduler", "cosine",
            "--weight_decay", "1e-3",
            "--mixup_alpha", "0.2",
            "--patience", "10",
            "--atc_n_windows", "7",
        ],
    },
]


def _latest_bcic_dir(before):
    after = {
        name for name in os.listdir(RESULTS_DIR)
        if name.startswith("BCIC2A_ATCNet_")
    }
    new_dirs = sorted(after - before)
    if not new_dirs:
        return None
    return os.path.join(RESULTS_DIR, new_dirs[-1])


def _load_metrics(run_dir):
    metrics_path = os.path.join(run_dir, "metrics.json")
    config_path = os.path.join(run_dir, "config.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return metrics, config


def _write_summary(rows):
    lines = [
        "# BCIC2A Round 3 Summary",
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

        run_dir = _latest_bcic_dir(before)
        if run_dir is None:
            print(f"[WARN] Could not determine result directory for {exp['name']}")
            continue

        metrics, _ = _load_metrics(run_dir)
        row = {
            "name": exp["name"],
            "run_dir": run_dir,
            "best_val_accuracy": metrics["best_val_accuracy"],
            "best_epoch": metrics["best_epoch"],
        }
        rows.append(row)
        _write_summary(rows)

        print(
            f"Best val acc for {exp['name']}: "
            f"{metrics['best_val_accuracy'] * 100:.2f}%"
        )
        if metrics["best_val_accuracy"] >= 0.70:
            print("Reached 70% stop condition.")
            break


if __name__ == "__main__":
    main()

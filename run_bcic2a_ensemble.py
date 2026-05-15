"""Build a soft-voting ensemble from existing BCIC2A ATCNet runs.

This script averages class probabilities from multiple saved run directories,
reports validation accuracy on the corresponding split, and writes an ensemble
result directory with predictions.txt for the test set.
"""

import argparse
import json
import os
from datetime import datetime

import torch

from model import MODEL_DICT
from utils import load_dataset_info, create_dataloaders, start_log, stop_log


def build_model(cfg):
    channels, num_classes, window_sec = load_dataset_info(cfg["dataset"])
    time_points = int(window_sec * 200)
    model_cls = MODEL_DICT[cfg["model"]]
    kwargs = dict(chans=channels, num_classes=num_classes, time_point=time_points)
    if cfg["model"] == "ATCNet":
        kwargs["n_windows"] = cfg.get("atc_n_windows", 5)
        kwargs["F1"] = cfg.get("atc_f1", 16)
        kwargs["d_model"] = cfg.get("atc_d_model", 32)
        kwargs["dropout_conv"] = cfg.get("atc_dropout_conv", 0.3)
        kwargs["dropout_attn"] = cfg.get("atc_dropout_attn", 0.5)
        kwargs["dropout_tcn"] = cfg.get("atc_dropout_tcn", 0.3)
        kwargs["tcn_depth"] = cfg.get("atc_tcn_depth", 2)
    return model_cls(**kwargs)


def load_cfg(run_dir):
    with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def collect_probs(run_dir):
    cfg = load_cfg(run_dir)
    _, val_loader, test_loader = create_dataloaders(
        cfg["dataset"],
        cfg["batch_size"],
        fold=cfg.get("fold"),
        standardize=cfg.get("standardize_inputs", False),
    )

    model = build_model(cfg)
    state = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    val_probs = []
    val_labels = []
    test_probs = []
    with torch.no_grad():
        for x, y in val_loader:
            val_probs.append(torch.softmax(model(x), dim=1))
            val_labels.append(y)
        for x in test_loader:
            test_probs.append(torch.softmax(model(x), dim=1))

    return (
        torch.cat(val_probs, dim=0),
        torch.cat(val_labels, dim=0),
        torch.cat(test_probs, dim=0),
        cfg,
    )


def main():
    parser = argparse.ArgumentParser(description="BCIC2A ATCNet soft-voting ensemble")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Result directories under Results/ to ensemble",
    )
    args = parser.parse_args()

    run_dirs = [os.path.join("Results", run) for run in args.runs]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("Results", f"BCIC2A_ATCNet_ensemble_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    tee = start_log(out_dir)

    bundles = [collect_probs(run_dir) for run_dir in run_dirs]
    cfg = bundles[0][3]
    val_labels = bundles[0][1]

    val_probs = sum(bundle[0] for bundle in bundles) / len(bundles)
    test_probs = sum(bundle[2] for bundle in bundles) / len(bundles)

    val_preds = val_probs.argmax(dim=1)
    test_preds = test_probs.argmax(dim=1)
    val_acc = (val_preds == val_labels).float().mean().item()

    print("Ensemble runs:")
    for run in args.runs:
        print(f"  - {run}")
    print(f"Validation accuracy: {val_acc:.4f}")

    with open(os.path.join(out_dir, "predictions.txt"), "w", encoding="utf-8") as f:
        for label in test_preds.tolist():
            f.write(f"{int(label)}\n")

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": cfg["dataset"],
                "model": "ATCNetEnsemble",
                "member_runs": args.runs,
                "fold": cfg.get("fold"),
            },
            f,
            indent=2,
        )

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_accuracy": val_acc,
                "final_val_accuracy": val_acc,
                "best_epoch": None,
            },
            f,
            indent=2,
        )

    stop_log(tee)
    print(f"Saved ensemble outputs to: {out_dir}")


if __name__ == "__main__":
    main()

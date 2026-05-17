"""Estimate 5-fold ensemble accuracy via cross-validation.

For each validation fold, predict using the 4 models that didn't train on that
fold. This gives an unbiased estimate of the ensemble's generalization accuracy.

Usage:
    python eval_5fold_ensemble.py Results/BCIC2A_ATCNet_20260517_044019_allfolds
"""

import json
import os
import sys

import torch

from model import MODEL_DICT
from utils import load_dataset_info, create_dataloaders


def build_model(cfg):
    channels, num_classes, window_sec = load_dataset_info(cfg["dataset"])
    time_points = int(window_sec * 200)
    model_cls = MODEL_DICT[cfg["model"]]
    kwargs = dict(chans=channels, num_classes=num_classes, time_point=time_points)
    if cfg["model"] == "ATCNet":
        kwargs["F1"] = cfg.get("atc_f1", 16)
        kwargs["d_model"] = cfg.get("atc_d_model", 32)
        kwargs["n_windows"] = cfg.get("atc_n_windows", 5)
        kwargs["dropout_conv"] = cfg.get("atc_dropout_conv", 0.3)
        kwargs["dropout_attn"] = cfg.get("atc_dropout_attn", 0.5)
        kwargs["dropout_tcn"] = cfg.get("atc_dropout_tcn", 0.3)
        kwargs["tcn_depth"] = cfg.get("atc_tcn_depth", 2)
    return model_cls(**kwargs)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    run_dir = sys.argv[1]
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)

    dataset = cfg.get("dataset", "BCIC2A")
    batch_size = cfg.get("batch_size", 32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load models for all 5 folds
    models = []
    for k in range(1, 6):
        model_path = os.path.join(run_dir, f"fold_{k}", "model.pt")
        if not os.path.exists(model_path):
            continue
        model = build_model(cfg)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    n_folds = len(models)
    if n_folds < 2:
        print("Need at least 2 folds")
        sys.exit(1)

    print(f"Loaded {n_folds} fold models")

    # Cross-validated ensemble: for each fold k, use the other n-1 models
    all_val_preds = []
    all_val_labels = []
    val_accs = []

    for k in range(1, n_folds + 1):
        _, val_loader, _ = create_dataloaders(
            dataset, batch_size, fold=k,
            standardize=cfg.get("standardize_inputs", False),
        )

        # Use all models EXCEPT fold k
        ensemble_models = [m for i, m in enumerate(models) if i != k - 1]

        fold_correct = 0
        fold_total = 0

        with torch.no_grad():
            for batch in val_loader:
                data, label = batch[0], batch[1]
                data = data.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                # Average probabilities from all held-in models
                probs = 0
                for m in ensemble_models:
                    probs += torch.softmax(m(data), dim=1)
                probs /= len(ensemble_models)

                preds = probs.argmax(dim=1)
                fold_correct += (preds == label).sum().item()
                fold_total += label.size(0)

        fold_acc = fold_correct / fold_total
        val_accs.append(fold_acc)
        print(f"  Fold {k} CV ensemble: {fold_acc * 100:.2f}%")

    mean_acc = sum(val_accs) / len(val_accs)
    print(f"\n5-fold CV ensemble accuracy: {mean_acc * 100:.2f}%")
    print(f"Per-fold: {[f'{a * 100:.2f}%' for a in val_accs]}")


if __name__ == "__main__":
    main()

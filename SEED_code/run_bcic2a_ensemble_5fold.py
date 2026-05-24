"""Ensemble 5-fold CV models and predict on the test set.

Usage:
    python run_bcic2a_ensemble_5fold.py Results/BCIC2A_ATCNet_20260517_044019_allfolds
"""

import json
import os
import sys
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
        kwargs["F1"] = cfg.get("atc_f1", 16)
        kwargs["d_model"] = cfg.get("atc_d_model", 32)
        kwargs["n_windows"] = cfg.get("atc_n_windows", 5)
        kwargs["dropout_conv"] = cfg.get("atc_dropout_conv", 0.3)
        kwargs["dropout_attn"] = cfg.get("atc_dropout_attn", 0.3)
        kwargs["attn_drop"] = cfg.get("atc_attn_drop", 0.5)
        kwargs["dropout_tcn"] = cfg.get("atc_dropout_tcn", 0.3)
        kwargs["tcn_depth"] = cfg.get("atc_tcn_depth", 2)
        kwargs["residual_drop"] = cfg.get("atc_residual_drop", 0.0)
        kwargs["drop_path_prob"] = cfg.get("atc_drop_path_prob", 0.0)
    elif cfg["model"] == "EEGConformer":
        kwargs["dim"] = cfg.get("conf_dim", 64)
        kwargs["n_blocks"] = cfg.get("conf_blocks", 4)
        kwargs["n_head"] = cfg.get("conf_heads", 4)
        kwargs["kernel_size"] = cfg.get("conf_kernel", 31)
        kwargs["ff_expansion"] = cfg.get("conf_ff_expansion", 4)
        kwargs["patch_kernel"] = cfg.get("conf_patch_kernel", 25)
        kwargs["patch_stride"] = cfg.get("conf_patch_stride", 10)
        kwargs["dropout"] = cfg.get("conf_dropout", 0.1)
    return model_cls(**kwargs)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    run_dir = sys.argv[1]
    if not os.path.isdir(run_dir):
        print(f"Error: {run_dir} not found")
        sys.exit(1)

    # Load config from parent directory
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found")
        sys.exit(1)
    with open(config_path) as f:
        cfg = json.load(f)

    dataset = cfg.get("dataset", "BCIC2A")
    batch_size = cfg.get("batch_size", 32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build test loader (no fold = original test set)
    _, _, test_loader = create_dataloaders(
        dataset,
        batch_size,
        fold=None,  # use original test_x_only.h5
        standardize=cfg.get("standardize_inputs", False),
    )

    model_cls = MODEL_DICT[cfg["model"]]
    fold_dirs = [
        os.path.join(run_dir, f"fold_{k}")
        for k in range(1, 6)
        if os.path.isdir(os.path.join(run_dir, f"fold_{k}"))
    ]

    if len(fold_dirs) < 5:
        print(f"Warning: found only {len(fold_dirs)} fold directories")
    print(f"Found {len(fold_dirs)} fold models in: {run_dir}")

    # Load each fold model and collect test probabilities
    all_test_probs = []
    device = torch.device(device)

    for fold_dir in fold_dirs:
        model_path = os.path.join(fold_dir, "model.pt")
        if not os.path.exists(model_path):
            print(f"  Skipping {fold_dir}: model.pt not found")
            continue

        model = build_model(cfg)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        test_probs = []
        with torch.no_grad():
            for x in test_loader:
                x = x.to(device, non_blocking=True)
                logits = model(x)
                test_probs.append(torch.softmax(logits, dim=1).cpu())

        fold_probs = torch.cat(test_probs, dim=0)
        all_test_probs.append(fold_probs)
        print(f"  Fold {os.path.basename(fold_dir)}: loaded ({fold_probs.size(0)} test samples)")

    if not all_test_probs:
        print("Error: no models loaded")
        sys.exit(1)

    # Average probabilities (soft voting)
    ensemble_probs = torch.stack(all_test_probs).mean(dim=0)
    ensemble_preds = ensemble_probs.argmax(dim=1)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        os.path.dirname(run_dir),
        f"BCIC2A_ATCNet_5fold_ensemble_{timestamp}",
    )
    os.makedirs(out_dir, exist_ok=True)
    tee = start_log(out_dir)

    output_path = os.path.join(out_dir, "predictions.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for label in ensemble_preds.tolist():
            f.write(f"{int(label)}\n")

    # Save ensemble config
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "dataset": dataset,
            "model": f"{cfg['model']}_5fold_ensemble",
            "member_folds": [os.path.basename(d) for d in fold_dirs],
            "parent_run": os.path.basename(run_dir),
        }, f, indent=2)

    print(f"\nSaved {len(ensemble_preds)} ensemble test predictions to: {output_path}")
    print(f"Output directory: {out_dir}")
    stop_log(tee)


if __name__ == "__main__":
    main()

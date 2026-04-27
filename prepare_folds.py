"""Generate stratified k-fold CV splits for a dataset.

Merges the original train.h5 and val.h5, saves the combined data to all.h5,
and generates stratified fold indices saved as .npy files.

Must be run once before using --fold in run_train.py.

Usage:
    python prepare_folds.py --dataset MDD
    python prepare_folds.py --dataset SLEEP --n_folds 5 --seed 42
"""

import argparse
import json
import os

import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold


def main():
    parser = argparse.ArgumentParser(description="Generate k-fold CV splits")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["MDD", "BCIC2A", "CHINESE", "SEED", "SLEEP"],
        help="Dataset name",
    )
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_dir = os.path.join("data", args.dataset)
    train_path = os.path.join(data_dir, "train.h5")
    val_path = os.path.join(data_dir, "val.h5")
    all_path = os.path.join(data_dir, "all.h5")

    # --- Load and merge train + val ---
    print(f"Loading {args.dataset}...")
    with h5py.File(train_path, "r") as f:
        X_train = f["X"][()]
        y_train = f["y"][()]
    with h5py.File(val_path, "r") as f:
        X_val = f["X"][()]
        y_val = f["y"][()]

    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)

    print(f"  Total samples: {len(X_all)}  Shape: {list(X_all.shape[1:])}")

    # Save merged all.h5 (overwrite if re-running)
    with h5py.File(all_path, "w") as f:
        f.create_dataset("X", data=X_all, dtype="float32")
        f.create_dataset("y", data=y_all, dtype="int64")
    print(f"  Saved: {all_path}")

    # --- Generate stratified folds ---
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    fold_info = {
        "dataset": args.dataset,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "total_samples": len(X_all),
        "shape": list(X_all.shape[1:]),
        "folds": {},
    }

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
        fold_dir = os.path.join(data_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        np.save(os.path.join(fold_dir, "train_idx.npy"), train_idx)
        np.save(os.path.join(fold_dir, "val_idx.npy"), val_idx)

        fold_info["folds"][str(fold_idx)] = {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
        }
        print(f"  Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}")

    with open(os.path.join(data_dir, "folds_info.json"), "w") as f:
        json.dump(fold_info, f, indent=2)

    print(f"\nDone. {args.n_folds} folds prepared under {data_dir}/fold_*/")


if __name__ == "__main__":
    main()

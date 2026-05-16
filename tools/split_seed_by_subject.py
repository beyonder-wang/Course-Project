#!/usr/bin/env python
"""Split existing SEED train/val/test by subject index.

The data is organized as:
  train.h5: 15 subjects × 60 segments = 900 total (labels balanced: 20 per class per subject)
  val.h5:   15 subjects × 30 segments = 450 total (labels balanced: 10 per class per subject)
  test_x_only.h5: 15 subjects × 30 segments = 450 total

Creates subject-based splits for cross-subject evaluation.
"""

import argparse
import json
import os

import h5py
import numpy as np


def _load_split(path, has_labels=True):
    with h5py.File(path, "r") as f:
        X = f["X"][()].astype(np.float32)
        y = f["y"][()].astype(np.int64) if has_labels and "y" in f else None
    return X, y


def _write_split(path, X, y=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X)
        if y is not None:
            f.create_dataset("y", data=y)


def main():
    parser = argparse.ArgumentParser(description="Split SEED data by subject")
    parser.add_argument("--data_dir", default="data/SEED")
    parser.add_argument("--target_dir", default="data/SEED_BYSUBJ")
    parser.add_argument("--train_subjects", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--val_subjects", type=int, nargs="+", default=list(range(13, 16)))
    parser.add_argument("--subjects_per_train_file", type=int, default=60)
    parser.add_argument("--subjects_per_val_file", type=int, default=30)
    args = parser.parse_args()

    train_X, train_y = _load_split(os.path.join(args.data_dir, "train.h5"))
    val_X, val_y = _load_split(os.path.join(args.data_dir, "val.h5"))
    test_X, _ = _load_split(os.path.join(args.data_dir, "test_x_only.h5"), has_labels=False)

    n_subjects = 15
    train_per_subj = args.subjects_per_train_file  # 60
    val_per_subj = args.subjects_per_val_file      # 30

    print(f"train.h5: {train_X.shape}, val.h5: {val_X.shape}, test_x_only.h5: {test_X.shape}")
    print(f"Train subjects: {args.train_subjects}")
    print(f"Val subjects: {args.val_subjects}")

    def _gather_subjects(X, y, subjects, per_subj):
        """Gather segments for given subjects from X (0-indexed subjects)."""
        indices = []
        for s in subjects:
            start = (s - 1) * per_subj
            end = start + per_subj
            indices.extend(range(start, end))
        if y is not None:
            return X[indices], y[indices]
        return X[indices], None

    # Build new train set: subjects from train.h5 + val.h5
    tr_X1, tr_y1 = _gather_subjects(train_X, train_y, args.train_subjects, train_per_subj)
    tr_X2, tr_y2 = _gather_subjects(val_X, val_y, args.train_subjects, val_per_subj)
    new_train_X = np.concatenate([tr_X1, tr_X2], axis=0)
    new_train_y = np.concatenate([tr_y1, tr_y2], axis=0)

    # Build new val set
    vl_X1, vl_y1 = _gather_subjects(train_X, train_y, args.val_subjects, train_per_subj)
    vl_X2, vl_y2 = _gather_subjects(val_X, val_y, args.val_subjects, val_per_subj)
    new_val_X = np.concatenate([vl_X1, vl_X2], axis=0)
    new_val_y = np.concatenate([vl_y1, vl_y2], axis=0)

    # Build new test set
    ts_X, _ = _gather_subjects(test_X, None, args.val_subjects, val_per_subj)
    new_test_X = ts_X

    print(f"\nNew train: {new_train_X.shape}, labels={dict(zip(*np.unique(new_train_y, return_counts=True)))}")
    print(f"New val:   {new_val_X.shape}, labels={dict(zip(*np.unique(new_val_y, return_counts=True)))}")
    print(f"New test:  {new_test_X.shape}")

    _write_split(os.path.join(args.target_dir, "train.h5"), new_train_X, new_train_y)
    _write_split(os.path.join(args.target_dir, "val.h5"), new_val_X, new_val_y)
    _write_split(os.path.join(args.target_dir, "test_x_only.h5"), new_test_X)

    # Copy dataset_info.json
    src_info = os.path.join(args.data_dir, "dataset_info.json")
    if os.path.exists(src_info):
        with open(src_info) as f:
            info = json.load(f)
        info["dataset"]["name"] = "SEED_BYSUBJ"
        with open(os.path.join(args.target_dir, "dataset_info.json"), "w") as f:
            json.dump(info, f, indent=2)

    print(f"\nDone: {args.target_dir}/")


if __name__ == "__main__":
    main()

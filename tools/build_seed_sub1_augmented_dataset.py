"""Build a SEED variant augmented with unseen full-data segments from sub_1.h5.

This script:
1. loads the current downsampled SEED train/val/test splits,
2. hashes every sample in those splits to identify already-used sub_1 segments,
3. collects all remaining labeled segments from ``data/SEED/SEED/sub_1.h5``,
4. appends those unseen sub_1 segments to the training set, and
5. removes subject 1 from validation to avoid validation leakage.

The original test split is copied through unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np


def _hash_array(x: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()


def _load_h5(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {key: f[key][()] for key in f.keys()}


def _write_h5(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=value)


def _collect_unseen_sub1_segments(sub1_path: Path, seen_hashes: set[str]) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    skipped = 0

    with h5py.File(sub1_path, "r") as f:
        trials = sorted(f.keys(), key=lambda name: int(name.replace("trial", "")))
        for trial_name in trials:
            trial = f[trial_name]
            for segment_name in sorted(trial.keys(), key=lambda name: int(name.replace("segment", ""))):
                eeg_ds = trial[segment_name]["eeg"]
                eeg = eeg_ds[()].astype(np.float32)
                sample_hash = _hash_array(eeg)
                if sample_hash in seen_hashes:
                    skipped += 1
                    continue
                label = int(np.asarray(eeg_ds.attrs["label"]).reshape(-1)[0])
                xs.append(eeg)
                ys.append(label)

    if not xs:
        raise RuntimeError("No unseen sub_1 segments were found to augment the training set.")

    print(f"[collect] unseen sub_1 segments: {len(xs)} (skipped already-used: {skipped})")
    return np.stack(xs, axis=0).astype(np.float32), np.asarray(ys, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SEED augmented with unseen sub_1 full-data samples")
    parser.add_argument("--source_dir", default="data/SEED")
    parser.add_argument("--sub1_path", default="data/SEED/SEED/sub_1.h5")
    parser.add_argument("--target_dir", default="data/SEED_SUB1FULL_AUG")
    parser.add_argument("--val_subject_to_drop", type=int, default=1)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    sub1_path = Path(args.sub1_path)

    train = _load_h5(source_dir / "train.h5")
    val = _load_h5(source_dir / "val.h5")
    test = _load_h5(source_dir / "test_x_only.h5")

    if "subject_id" not in train or "subject_id" not in val:
        raise ValueError("Source SEED train/val must already contain subject_id metadata.")

    seen_hashes = set()
    for split_name, arrays in (("train", train), ("val", val), ("test", test)):
        for x in arrays["X"]:
            seen_hashes.add(_hash_array(x.astype(np.float32)))
        print(f"[hash] {split_name}: {len(arrays['X'])} samples")

    extra_x, extra_y = _collect_unseen_sub1_segments(sub1_path, seen_hashes)
    extra_subject_id = np.full((len(extra_x),), int(args.val_subject_to_drop), dtype=np.int64)

    new_train = {
        "X": np.concatenate([train["X"].astype(np.float32), extra_x], axis=0),
        "y": np.concatenate([train["y"].astype(np.int64), extra_y], axis=0),
        "subject_id": np.concatenate([train["subject_id"].astype(np.int64), extra_subject_id], axis=0),
    }

    keep_mask = val["subject_id"].astype(np.int64) != int(args.val_subject_to_drop)
    new_val = {
        "X": val["X"][keep_mask].astype(np.float32),
        "y": val["y"][keep_mask].astype(np.int64),
        "subject_id": val["subject_id"][keep_mask].astype(np.int64),
    }

    print(f"[train] original={len(train['X'])} -> augmented={len(new_train['X'])}")
    print(f"[val] original={len(val['X'])} -> filtered={len(new_val['X'])}")
    print(
        "[val] labels after drop:",
        {int(k): int(v) for k, v in zip(*np.unique(new_val["y"], return_counts=True))},
    )

    _write_h5(target_dir / "train.h5", new_train)
    _write_h5(target_dir / "val.h5", new_val)
    shutil.copy2(source_dir / "test_x_only.h5", target_dir / "test_x_only.h5")

    info_path = source_dir / "dataset_info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    info["dataset"]["name"] = "SEED_SUB1FULL_AUG"
    info["dataset"]["description"] = (
        "SEED downsampled split augmented with unseen full-data sub_1 segments; "
        "validation excludes subject 1."
    )
    info.setdefault("augmentation", {})
    info["augmentation"]["sub1_unseen_train_samples"] = int(len(extra_x))
    info["augmentation"]["dropped_val_subject"] = int(args.val_subject_to_drop)
    info["augmentation"]["train_total"] = int(len(new_train["X"]))
    info["augmentation"]["val_total"] = int(len(new_val["X"]))
    with (target_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"[done] wrote augmented dataset to {target_dir}")


if __name__ == "__main__":
    main()

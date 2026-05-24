"""Build a DE+LDS augmented dataset from SEED_DE_INIT plus unseen full-data sub_1.

This mirrors the raw ``SEED_SUB1FULL_AUG`` construction, but keeps everything in
DE+LDS feature space:

- base train/val/test come from ``data/SEED_DE_INIT``
- unseen segments come from full ``sub_1.h5`` with full-trial LDS smoothing
- validation drops subject 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np


BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 75.0),
]


def _hash_array(x: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()


def _fft_band_de(eeg: np.ndarray, fs: float) -> np.ndarray:
    x_fft = np.fft.rfft(eeg, axis=-1)
    freqs = np.fft.rfftfreq(eeg.shape[-1], d=1.0 / fs)
    features = []
    for _, low, high in BANDS:
        mask = (freqs >= low) & (freqs <= high)
        filtered = np.fft.irfft(x_fft * mask[None, :], n=eeg.shape[-1], axis=-1)
        var = filtered.var(axis=-1) + 1e-6
        features.append(0.5 * np.log(var))
    return np.stack(features, axis=-1).astype(np.float32)


def _lds_smooth_sequence(seq: np.ndarray, process_noise: float = 1e-4, measurement_noise: float = 1e-2) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    t_steps, feat_dim = seq.shape
    filt_mean = np.zeros_like(seq)
    filt_var = np.zeros((t_steps, feat_dim), dtype=np.float32)

    filt_mean[0] = seq[0]
    filt_var[0] = 1.0
    for t in range(1, t_steps):
        pred_mean = filt_mean[t - 1]
        pred_var = filt_var[t - 1] + process_noise
        kalman_gain = pred_var / (pred_var + measurement_noise)
        filt_mean[t] = pred_mean + kalman_gain * (seq[t] - pred_mean)
        filt_var[t] = (1.0 - kalman_gain) * pred_var

    smooth_mean = filt_mean.copy()
    smooth_var = filt_var.copy()
    for t in range(t_steps - 2, -1, -1):
        pred_var = filt_var[t] + process_noise
        gain = filt_var[t] / pred_var
        smooth_mean[t] = filt_mean[t] + gain * (smooth_mean[t + 1] - filt_mean[t])
        smooth_var[t] = filt_var[t] + gain * (smooth_var[t + 1] - pred_var) * gain
    return smooth_mean


def _load_h5(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {key: f[key][()] for key in f.keys()}


def _write_h5(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=value)


def _collect_unseen_sub1_de(
    sub1_path: Path,
    seen_hashes: set[str],
    fs: float,
    process_noise: float,
    measurement_noise: float,
) -> dict[str, np.ndarray]:
    xs = []
    ys = []
    subject_ids = []
    trial_ids = []
    segment_ids = []
    skipped = 0

    with h5py.File(sub1_path, "r") as f:
        trials = sorted(f.keys(), key=lambda name: int(name.replace("trial", "")))
        for trial_name in trials:
            trial = f[trial_name]
            trial_id = int(trial.attrs["trial_id"])

            ordered_segments = []
            for seg_name in trial.keys():
                eeg_ds = trial[seg_name]["eeg"]
                eeg = eeg_ds[()].astype(np.float32)
                label = int(np.asarray(eeg_ds.attrs["label"]).reshape(-1)[0])
                segment_id = int(eeg_ds.attrs["segment_id"])
                ordered_segments.append(
                    {
                        "hash": _hash_array(eeg),
                        "eeg": eeg,
                        "label": label,
                        "segment_id": segment_id,
                    }
                )

            ordered_segments.sort(key=lambda item: item["segment_id"])
            de_stack = np.stack([_fft_band_de(item["eeg"], fs=fs) for item in ordered_segments], axis=0)
            smooth_stack = _lds_smooth_sequence(
                de_stack.reshape(de_stack.shape[0], -1),
                process_noise=process_noise,
                measurement_noise=measurement_noise,
            ).reshape(de_stack.shape)

            for item, smooth_x in zip(ordered_segments, smooth_stack):
                if item["hash"] in seen_hashes:
                    skipped += 1
                    continue
                xs.append(smooth_x.astype(np.float32))
                ys.append(item["label"])
                subject_ids.append(1)
                trial_ids.append(trial_id)
                segment_ids.append(item["segment_id"])

    print(f"[collect] unseen sub_1 DE+LDS segments: {len(xs)} (skipped already-used: {skipped})")
    return {
        "X": np.stack(xs, axis=0).astype(np.float32),
        "y": np.asarray(ys, dtype=np.int64),
        "subject_id": np.asarray(subject_ids, dtype=np.int64),
        "trial_id": np.asarray(trial_ids, dtype=np.int64),
        "segment_id": np.asarray(segment_ids, dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SEED_SUB1FULL_AUG_DE from SEED_DE_INIT plus full sub_1")
    parser.add_argument("--base_raw_dir", default="data/SEED")
    parser.add_argument("--base_de_dir", default="data/SEED_DE_INIT")
    parser.add_argument("--sub1_path", default="data/SEED/SEED/sub_1.h5")
    parser.add_argument("--target_dir", default="data/SEED_SUB1FULL_AUG_DE")
    parser.add_argument("--drop_val_subject", type=int, default=1)
    parser.add_argument("--fs", type=float, default=200.0)
    parser.add_argument("--process_noise", type=float, default=1e-4)
    parser.add_argument("--measurement_noise", type=float, default=1e-2)
    args = parser.parse_args()

    base_raw_dir = Path(args.base_raw_dir)
    base_de_dir = Path(args.base_de_dir)
    target_dir = Path(args.target_dir)
    sub1_path = Path(args.sub1_path)

    base_train = _load_h5(base_de_dir / "train.h5")
    base_val = _load_h5(base_de_dir / "val.h5")

    seen_hashes = set()
    for split_name in ("train.h5", "val.h5", "test_x_only.h5"):
        with h5py.File(base_raw_dir / split_name, "r") as f:
            for x in f["X"][()]:
                seen_hashes.add(_hash_array(x.astype(np.float32)))
        print(f"[hash] {split_name}")

    extra = _collect_unseen_sub1_de(
        sub1_path=sub1_path,
        seen_hashes=seen_hashes,
        fs=args.fs,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
    )

    new_train = {
        "X": np.concatenate([base_train["X"].astype(np.float32), extra["X"]], axis=0),
        "y": np.concatenate([base_train["y"].astype(np.int64), extra["y"]], axis=0),
        "subject_id": np.concatenate([base_train["subject_id"].astype(np.int64), extra["subject_id"]], axis=0),
        "trial_id": np.concatenate([base_train["trial_id"].astype(np.int64), extra["trial_id"]], axis=0),
        "segment_id": np.concatenate([base_train["segment_id"].astype(np.int64), extra["segment_id"]], axis=0),
    }

    keep_mask = base_val["subject_id"].astype(np.int64) != int(args.drop_val_subject)
    new_val = {
        "X": base_val["X"][keep_mask].astype(np.float32),
        "y": base_val["y"][keep_mask].astype(np.int64),
        "subject_id": base_val["subject_id"][keep_mask].astype(np.int64),
        "trial_id": base_val["trial_id"][keep_mask].astype(np.int64),
        "segment_id": base_val["segment_id"][keep_mask].astype(np.int64),
    }

    print(f"[train] {len(base_train['X'])} -> {len(new_train['X'])}")
    print(f"[val] {len(base_val['X'])} -> {len(new_val['X'])}")

    _write_h5(target_dir / "train.h5", new_train)
    _write_h5(target_dir / "val.h5", new_val)
    shutil.copy2(base_de_dir / "test_x_only.h5", target_dir / "test_x_only.h5")

    with (base_de_dir / "dataset_info.json").open("r", encoding="utf-8") as f:
        info = json.load(f)
    info["dataset"]["name"] = "SEED_SUB1FULL_AUG_DE"
    info["dataset"]["description"] = (
        "SEED_DE_INIT augmented with unseen full-data sub_1 DE+LDS segments; validation excludes subject 1."
    )
    info.setdefault("augmentation", {})
    info["augmentation"]["sub1_unseen_train_samples"] = int(len(extra["X"]))
    info["augmentation"]["dropped_val_subject"] = int(args.drop_val_subject)
    info["augmentation"]["train_total"] = int(len(new_train["X"]))
    info["augmentation"]["val_total"] = int(len(new_val["X"]))
    with (target_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"[done] wrote dataset to {target_dir}")


if __name__ == "__main__":
    main()

"""Build a DE+LDS feature dataset from the current downsampled SEED split.

This uses the teacher-provided ``train_idx.txt`` / ``val_idx.txt`` to recover
subject / trial / segment ordering for the current train and val splits, then:

1. computes 5-band differential entropy per segment from the raw waveform, and
2. applies LDS-style smoothing within each available (subject, trial) subsequence.

Because no aligned test index file is available, ``test_x_only.h5`` is converted
to per-sample DE features without LDS smoothing.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
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

SUBJECT_RE = re.compile(r"sub_(\d+)\.h5")
TRIAL_RE = re.compile(r"trial(\d+)")
SEGMENT_RE = re.compile(r"segment(\d+)")

SEED_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2",
    "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5",
    "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1",
    "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6",
    "PO8", "CB1", "O1", "OZ", "O2", "CB2",
]


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


def _load_split(path: Path, has_labels: bool) -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(path, "r") as f:
        x = f["X"][()].astype(np.float32)
        y = f["y"][()].astype(np.int64) if has_labels and "y" in f else None
    return x, y


def _parse_index_lines(path: Path) -> list[dict[str, int]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            sample_path, label_text = line.rsplit(",", 1)
            file_part, trail_part = sample_path.split(",", 1)

            subject_match = SUBJECT_RE.search(file_part)
            trial_match = TRIAL_RE.search(trail_part)
            segment_match = SEGMENT_RE.search(trail_part)
            if subject_match is None or trial_match is None or segment_match is None:
                raise ValueError(f"Could not parse subject/trial/segment from {path}:{lineno}: {line!r}")

            rows.append(
                {
                    "subject_id": int(subject_match.group(1)),
                    "trial_id": int(trial_match.group(1)),
                    "segment_id": int(segment_match.group(1)),
                    "label": int(label_text),
                }
            )
    return rows


def _build_supervised_split(
    split_name: str,
    x: np.ndarray,
    y: np.ndarray,
    rows: list[dict[str, int]],
    fs: float,
    process_noise: float,
    measurement_noise: float,
) -> dict[str, np.ndarray]:
    if len(x) != len(rows):
        raise ValueError(f"{split_name}: x length {len(x)} != metadata rows {len(rows)}")

    txt_y = np.asarray([row["label"] for row in rows], dtype=np.int64)
    if not np.array_equal(y, txt_y):
        mismatch = int(np.flatnonzero(y != txt_y)[0])
        raise ValueError(
            f"{split_name}: labels mismatch at index {mismatch}: h5={int(y[mismatch])} txt={int(txt_y[mismatch])}"
        )

    out_x = np.zeros((len(x), x.shape[1], len(BANDS)), dtype=np.float32)
    subject_ids = np.asarray([row["subject_id"] for row in rows], dtype=np.int64)
    trial_ids = np.asarray([row["trial_id"] for row in rows], dtype=np.int64)
    segment_ids = np.asarray([row["segment_id"] for row in rows], dtype=np.int64)

    grouped_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped_indices[(row["subject_id"], row["trial_id"])].append(idx)

    for (_, _), indices in grouped_indices.items():
        indices = sorted(indices, key=lambda idx: rows[idx]["segment_id"])
        de_stack = np.stack([_fft_band_de(x[idx], fs=fs) for idx in indices], axis=0)
        smooth_stack = _lds_smooth_sequence(
            de_stack.reshape(de_stack.shape[0], -1),
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        ).reshape(de_stack.shape)
        for out_idx, smooth_x in zip(indices, smooth_stack):
            out_x[out_idx] = smooth_x.astype(np.float32)

    print(f"[{split_name}] samples={len(x)} groups={len(grouped_indices)}")
    return {
        "X": out_x,
        "y": y.astype(np.int64),
        "subject_id": subject_ids,
        "trial_id": trial_ids,
        "segment_id": segment_ids,
    }


def _build_test_split(x: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    out_x = np.stack([_fft_band_de(sample, fs=fs) for sample in x], axis=0).astype(np.float32)
    print(f"[test] samples={len(x)} (DE only, no LDS metadata available)")
    return {"X": out_x}


def _write_h5(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=value)


def _write_dataset_info(target_dir: Path) -> None:
    info = {
        "dataset": {
            "name": "SEED_DE_INIT",
            "description": "DE+LDS features rebuilt from the current downsampled SEED split",
            "task_type": "emotion",
            "downstream_task": "classification",
            "num_labels": 3,
            "category_list": ["negative", "neutral", "positive"],
            "channels": SEED_CHANNELS,
            "montage": "10_20",
            "source_url": "https://bcmi.sjtu.edu.cn/home/seed/seed.html",
        },
        "processing": {
            "target_sampling_rate": 200.0,
            "window_sec": 0.025,
            "feature_type": "de_lds_partial",
            "feature_bands": [name for name, _, _ in BANDS],
        },
    }
    with (target_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DE+LDS dataset from current SEED split")
    parser.add_argument("--source_dir", default="data/SEED")
    parser.add_argument("--target_dir", default="data/SEED_DE_INIT")
    parser.add_argument("--fs", type=float, default=200.0)
    parser.add_argument("--process_noise", type=float, default=1e-4)
    parser.add_argument("--measurement_noise", type=float, default=1e-2)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)

    train_x, train_y = _load_split(source_dir / "train.h5", has_labels=True)
    val_x, val_y = _load_split(source_dir / "val.h5", has_labels=True)
    test_x, _ = _load_split(source_dir / "test_x_only.h5", has_labels=False)

    train_rows = _parse_index_lines(source_dir / "train_idx.txt")
    val_rows = _parse_index_lines(source_dir / "val_idx.txt")

    train_arrays = _build_supervised_split(
        "train", train_x, train_y, train_rows, fs=args.fs,
        process_noise=args.process_noise, measurement_noise=args.measurement_noise,
    )
    val_arrays = _build_supervised_split(
        "val", val_x, val_y, val_rows, fs=args.fs,
        process_noise=args.process_noise, measurement_noise=args.measurement_noise,
    )
    test_arrays = _build_test_split(test_x, fs=args.fs)

    _write_h5(target_dir / "train.h5", train_arrays)
    _write_h5(target_dir / "val.h5", val_arrays)
    _write_h5(target_dir / "test_x_only.h5", test_arrays)
    _write_dataset_info(target_dir)
    print(f"[done] wrote dataset to {target_dir}")


if __name__ == "__main__":
    main()

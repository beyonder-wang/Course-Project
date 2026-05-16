#!/usr/bin/env python
"""Build SEED_SUB1_DE dataset from sub_1.h5 with proper DE+LDS features.

Uses only subject 1 data (the only subject file available).
Splits by session: session 1+2 -> train, session 3 -> val.
Computes 5-band DE features with LDS (Kalman) temporal smoothing within each trial.
"""

import argparse
import os
import sys

import h5py
import numpy as np

BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 75.0),
]


def fft_band_de(eeg, fs):
    """Compute 5-band differential entropy features from raw EEG.

    Args:
        eeg: (C, T) raw EEG
        fs: sampling rate

    Returns:
        (C, 5) DE features (log-variance per band)
    """
    x_fft = np.fft.rfft(eeg, axis=-1)
    freqs = np.fft.rfftfreq(eeg.shape[-1], d=1.0 / fs)
    features = []
    for _, low, high in BANDS:
        mask = (freqs >= low) & (freqs <= high)
        filtered = np.fft.irfft(x_fft * mask[None, :], n=eeg.shape[-1], axis=-1)
        var = filtered.var(axis=-1) + 1e-6
        features.append(0.5 * np.log(var))
    return np.stack(features, axis=-1).astype(np.float32)  # (C, 5)


def lds_smooth_sequence(seq, process_noise=1e-4, measurement_noise=1e-2):
    """Kalman RTS smoother for a 1D LDS with A=1, C=1, vectorized over features.

    Args:
        seq: (T, D) feature sequence
        process_noise: Q in Kalman filter
        measurement_noise: R in Kalman filter

    Returns:
        (T, D) smoothed sequence
    """
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


def process_trial(trial_group, fs, process_noise, measurement_noise):
    """Process one trial: sort segments, compute DE, apply LDS smoothing.

    Args:
        trial_group: h5py Group for one trial
        fs: sampling rate
        process_noise, measurement_noise: LDS parameters

    Returns:
        list of dicts with keys: X (C,5), y, segment_id, start_time, end_time
    """
    segments = []
    for seg_name in trial_group.keys():
        eeg_ds = trial_group[seg_name]["eeg"]
        eeg = eeg_ds[()].astype(np.float32)  # (C, T)
        label = int(np.asarray(eeg_ds.attrs["label"]).reshape(-1)[0])
        segment_id = int(eeg_ds.attrs["segment_id"])
        start_time = float(eeg_ds.attrs["start_time"])
        end_time = float(eeg_ds.attrs["end_time"])
        segments.append({
            "eeg": eeg,
            "label": label,
            "segment_id": segment_id,
            "start_time": start_time,
            "end_time": end_time,
        })

    # Sort by segment_id (temporal order within trial)
    segments.sort(key=lambda item: item["segment_id"])

    # Compute DE features
    de_stack = np.stack([fft_band_de(item["eeg"], fs=fs) for item in segments], axis=0)

    # LDS smooth within trial
    smooth_stack = lds_smooth_sequence(
        de_stack.reshape(de_stack.shape[0], -1),
        process_noise=process_noise,
        measurement_noise=measurement_noise,
    ).reshape(de_stack.shape)

    # Combine
    results = []
    for item, x_de in zip(segments, smooth_stack):
        results.append({
            "X": x_de.astype(np.float32),  # (C, 5)
            "y": item["label"],
            "segment_id": item["segment_id"],
            "start_time": item["start_time"],
            "end_time": item["end_time"],
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Build SEED_SUB1_DE from sub_1.h5")
    parser.add_argument("--source", default="data/SEED/SEED/sub_1.h5")
    parser.add_argument("--target_dir", default="data/SEED_SUB1_DE")
    parser.add_argument("--fs", type=float, default=200.0)
    parser.add_argument("--process_noise", type=float, default=1e-4)
    parser.add_argument("--measurement_noise", type=float, default=1e-2)
    parser.add_argument("--train_sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--val_sessions", type=int, nargs="+", default=[3])
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"ERROR: {args.source} not found", file=sys.stderr)
        sys.exit(1)

    print(f"[load] {args.source}")
    train_entries = []
    val_entries = []
    trial_count = 0

    with h5py.File(args.source, "r") as f:
        trials = sorted(f.keys(), key=lambda x: int(x.replace("trial", "")))
        for tname in trials:
            trial = f[tname]
            session_id = int(trial.attrs["session_id"])
            trial_id = int(trial.attrs["trial_id"])

            results = process_trial(
                trial, args.fs, args.process_noise, args.measurement_noise
            )

            for r in results:
                r["session_id"] = session_id
                r["trial_id"] = trial_id

            if session_id in args.train_sessions:
                train_entries.extend(results)
            elif session_id in args.val_sessions:
                val_entries.extend(results)

            trial_count += 1
            if trial_count % 10 == 0:
                print(f"  processed {trial_count} trials...")

    print(f"[done] train: {len(train_entries)} segments, val: {len(val_entries)} segments")

    # Write train.h5
    os.makedirs(args.target_dir, exist_ok=True)
    _write_split(os.path.join(args.target_dir, "train.h5"), train_entries)
    _write_split(os.path.join(args.target_dir, "val.h5"), val_entries)

    # Print label distribution
    train_labels = np.array([e["y"] for e in train_entries])
    val_labels = np.array([e["y"] for e in val_entries])
    print(f"[labels] train: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    print(f"[labels] val:   {dict(zip(*np.unique(val_labels, return_counts=True)))}")

    # Create minimal dataset_info.json
    import json
    info = {
        "dataset": {
            "name": "SEED_SUB1_DE",
            "description": "SEED subject 1 DE+LDS features, session-based split",
            "task_type": "emotion",
            "downstream_task": "classification",
            "num_labels": 3,
            "category_list": ["negative", "neutral", "positive"],
            "channels": 62,
            "feature_dim": 5,
            "feature_type": "de_lds",
            "feature_bands": [name for name, _, _ in BANDS],
            "original_sampling_rate": 200.0,
            "window_sec": 2.0,
            "num_train": len(train_entries),
            "num_val": len(val_entries),
        }
    }
    with open(os.path.join(args.target_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"[done] dataset written to {args.target_dir}")


def _write_split(path, entries):
    """Write entries to h5 file."""
    X = np.stack([e["X"] for e in entries], axis=0).astype(np.float32)
    y = np.array([e["y"] for e in entries], dtype=np.int64)
    session_id = np.array([e["session_id"] for e in entries], dtype=np.int64)
    trial_id = np.array([e["trial_id"] for e in entries], dtype=np.int64)
    segment_id = np.array([e["segment_id"] for e in entries], dtype=np.int64)
    start_time = np.array([e["start_time"] for e in entries], dtype=np.float32)
    end_time = np.array([e["end_time"] for e in entries], dtype=np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)
        f.create_dataset("session_id", data=session_id)
        f.create_dataset("trial_id", data=trial_id)
        f.create_dataset("segment_id", data=segment_id)
        f.create_dataset("start_time", data=start_time)
        f.create_dataset("end_time", data=end_time)


if __name__ == "__main__":
    main()

"""Build SEED DE-feature splits aligned to the current raw train/val/test split."""

import argparse
import glob
import hashlib
import json
import os

import h5py
import numpy as np


BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 75.0),
]


def _hash_array(x):
    return hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()


def _fft_band_de(eeg, fs):
    x_fft = np.fft.rfft(eeg, axis=-1)
    freqs = np.fft.rfftfreq(eeg.shape[-1], d=1.0 / fs)
    features = []
    for _, low, high in BANDS:
        mask = (freqs >= low) & (freqs <= high)
        filtered = np.fft.irfft(x_fft * mask[None, :], n=eeg.shape[-1], axis=-1)
        var = filtered.var(axis=-1) + 1e-6
        features.append(0.5 * np.log(var))
    return np.stack(features, axis=-1).astype(np.float32)


def _lds_smooth_sequence(seq, process_noise=1e-4, measurement_noise=1e-2):
    """Kalman RTS smoother for a 1D LDS with A=1, C=1, vectorized over features."""
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


def _collect_entries(sub_path, subject_id, fs, process_noise, measurement_noise):
    entries = {}
    with h5py.File(sub_path, "r") as f:
        for trial_name in sorted(f.keys(), key=lambda x: int(x.replace("trial", ""))):
            trial = f[trial_name]
            session_id = int(trial.attrs["session_id"])
            trial_id = int(trial.attrs["trial_id"])

            segments = []
            for seg_name in trial.keys():
                eeg_ds = trial[seg_name]["eeg"]
                eeg = eeg_ds[()].astype(np.float32)
                label = int(np.asarray(eeg_ds.attrs["label"]).reshape(-1)[0])
                segment_id = int(eeg_ds.attrs["segment_id"])
                start_time = float(eeg_ds.attrs["start_time"])
                end_time = float(eeg_ds.attrs["end_time"])
                segments.append({
                    "hash": _hash_array(eeg),
                    "eeg": eeg,
                    "label": label,
                    "segment_id": segment_id,
                    "start_time": start_time,
                    "end_time": end_time,
                })

            segments.sort(key=lambda item: (item["segment_id"], item["start_time"]))
            de_stack = np.stack([_fft_band_de(item["eeg"], fs=fs) for item in segments], axis=0)
            smooth_stack = _lds_smooth_sequence(
                de_stack.reshape(de_stack.shape[0], -1),
                process_noise=process_noise,
                measurement_noise=measurement_noise,
            ).reshape(de_stack.shape)

            for item, x_de in zip(segments, smooth_stack):
                entries[item["hash"]] = {
                    "X": x_de.astype(np.float32),
                    "y": item["label"],
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "trial_id": trial_id,
                    "segment_id": item["segment_id"],
                    "start_time": item["start_time"],
                    "end_time": item["end_time"],
                }
    return entries


def _materialize_split(raw_split_path, entries_by_hash, has_labels=True):
    result = {
        "X": [],
        "y": [],
        "subject_id": [],
        "session_id": [],
        "trial_id": [],
        "segment_id": [],
        "start_time": [],
        "end_time": [],
    }
    matched = 0
    missed = 0
    with h5py.File(raw_split_path, "r") as f:
        X = f["X"]
        y = f["y"][()] if has_labels and "y" in f else None
        for idx in range(len(X)):
            raw_x = X[idx]
            item = entries_by_hash.get(_hash_array(raw_x))
            if item is None:
                missed += 1
                continue
            matched += 1
            result["X"].append(item["X"])
            if has_labels:
                split_label = int(y[idx])
                if split_label != item["y"]:
                    raise ValueError(
                        f"Label mismatch at {raw_split_path} index {idx}: split={split_label} map={item['y']}"
                    )
                result["y"].append(split_label)
            for key in ("subject_id", "session_id", "trial_id", "segment_id", "start_time", "end_time"):
                result[key].append(item[key])

    arrays = {
        "X": np.stack(result["X"], axis=0).astype(np.float32) if result["X"] else np.empty((0, 62, 5), dtype=np.float32),
        "subject_id": np.asarray(result["subject_id"], dtype=np.int64),
        "session_id": np.asarray(result["session_id"], dtype=np.int64),
        "trial_id": np.asarray(result["trial_id"], dtype=np.int64),
        "segment_id": np.asarray(result["segment_id"], dtype=np.int64),
        "start_time": np.asarray(result["start_time"], dtype=np.float32),
        "end_time": np.asarray(result["end_time"], dtype=np.float32),
    }
    if has_labels:
        arrays["y"] = np.asarray(result["y"], dtype=np.int64)
    return arrays, matched, missed


def _write_h5(path, arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=value)


def _write_dataset_info(target_dir):
    info = {
        "dataset": {
            "name": "SEED_DE",
            "description": "SEED DE-feature dataset aligned to the repo split",
            "task_type": "emotion",
            "downstream_task": "classification",
            "num_labels": 3,
            "category_list": ["negative", "neutral", "positive"],
            "original_sampling_rate": 1000.0,
            "channels": [
                "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2",
                "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
                "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5",
                "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1",
                "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6",
                "PO8", "CB1", "O1", "OZ", "O2", "CB2",
            ],
            "montage": "10_20",
            "source_url": "https://bcmi.sjtu.edu.cn/home/seed/seed.html",
        },
        "processing": {
            "target_sampling_rate": 200.0,
            "window_sec": 0.025,
            "feature_type": "de_lds_like",
            "feature_bands": [name for name, _, _ in BANDS],
        },
    }
    with open(os.path.join(target_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare SEED_DE splits from official-style subject files")
    parser.add_argument("--source_glob", default="data/SEED/SEED/sub_*.h5")
    parser.add_argument("--split_source_dir", default="data/SEED")
    parser.add_argument("--target_dir", default="data/SEED_DE")
    parser.add_argument("--fs", type=float, default=200.0)
    parser.add_argument("--process_noise", type=float, default=1e-4)
    parser.add_argument("--measurement_noise", type=float, default=1e-2)
    parser.add_argument("--allow_partial_match", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    subject_files = sorted(glob.glob(args.source_glob))
    if not subject_files:
        raise FileNotFoundError(f"No subject files matched: {args.source_glob}")

    entries_by_hash = {}
    for subject_idx, sub_path in enumerate(subject_files, start=1):
        print(f"[collect] subject {subject_idx}: {sub_path}")
        entries_by_hash.update(
            _collect_entries(
                sub_path,
                subject_id=subject_idx,
                fs=args.fs,
                process_noise=args.process_noise,
                measurement_noise=args.measurement_noise,
            )
        )
    print(f"[collect] cached segments: {len(entries_by_hash)}")

    splits = [
        ("train.h5", True),
        ("val.h5", True),
        ("test_x_only.h5", False),
    ]
    coverage = {}
    materialized = {}
    for filename, has_labels in splits:
        raw_path = os.path.join(args.split_source_dir, filename)
        arrays, matched, missed = _materialize_split(raw_path, entries_by_hash, has_labels=has_labels)
        coverage[filename] = {"matched": matched, "missed": missed}
        materialized[filename] = arrays
        print(f"[match] {filename}: matched={matched} missed={missed}")

    if not args.allow_partial_match:
        incomplete = {k: v for k, v in coverage.items() if v["missed"] > 0}
        if incomplete:
            subject_count = len(subject_files)
            coverage_lines = ", ".join(
                f"{name}: matched={stats['matched']} missed={stats['missed']}"
                for name, stats in coverage.items()
            )
            raise RuntimeError(
                "Split matching is incomplete. This is usually caused by an incomplete "
                f"`sub_*.h5` set rather than a Python version issue. Detected {subject_count} "
                f"subject file(s). Coverage: {coverage_lines}. Re-run with --allow_partial_match "
                "for inspection, or provide the full set of subject files."
            )

    if args.dry_run:
        return

    os.makedirs(args.target_dir, exist_ok=True)
    for filename, _ in splits:
        _write_h5(os.path.join(args.target_dir, filename), materialized[filename])
    _write_dataset_info(args.target_dir)

    with open(os.path.join(args.target_dir, "coverage.json"), "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2)
    print(f"[done] wrote DE-feature dataset to {args.target_dir}")


if __name__ == "__main__":
    main()

"""Attach SEED subject_id metadata from teacher-provided index text files.

The teacher-provided ``train_idx.txt`` / ``val_idx.txt`` files list each sample as::

    /zongsheng-group/SEED/sub_1.h5,trial9/segment22/eeg,2

The trailing integer is the class label. The subject/domain label is encoded in
the ``sub_*.h5`` path component. This script validates that the txt ordering
matches the current ``train.h5`` / ``val.h5`` labels, then writes a
``subject_id`` dataset into the HDF5 files so the existing subject-adversarial
training path can consume it directly.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np


SUBJECT_RE = re.compile(r"sub_(\d+)\.h5")


def _parse_index_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    subject_ids = []
    labels = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                sample_path, label_text = line.rsplit(",", 1)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno} is not 'sample_path,label': {line!r}") from exc

            match = SUBJECT_RE.search(sample_path)
            if match is None:
                raise ValueError(f"{path}:{lineno} does not contain sub_*.h5: {line!r}")

            subject_ids.append(int(match.group(1)))
            labels.append(int(label_text))

    return np.asarray(subject_ids, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def _validate_split(h5_path: Path, subject_ids: np.ndarray, txt_labels: np.ndarray) -> None:
    with h5py.File(h5_path, "r") as f:
        if "y" not in f:
            raise ValueError(f"{h5_path} is missing y; cannot validate txt alignment.")

        y = f["y"][()]
        if len(y) != len(subject_ids):
            raise ValueError(
                f"{h5_path} sample count mismatch: h5={len(y)} txt={len(subject_ids)}"
            )
        if not np.array_equal(y, txt_labels):
            mismatch = int(np.flatnonzero(y != txt_labels)[0])
            raise ValueError(
                f"{h5_path} label order mismatch at index {mismatch}: "
                f"h5={int(y[mismatch])} txt={int(txt_labels[mismatch])}"
            )


def _copy_h5_with_subject_id(src_path: Path, dst_path: Path, subject_ids: np.ndarray) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix=dst_path.stem + "_",
        suffix=".tmp.h5",
        dir=str(dst_path.parent),
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "w") as dst:
            sample_count = len(src["X"])
            if sample_count != len(subject_ids):
                raise ValueError(
                    f"{src_path} sample count mismatch while writing: "
                    f"h5={sample_count} subject_ids={len(subject_ids)}"
                )

            for key in src.keys():
                if key == "subject_id":
                    continue
                src.copy(src[key], dst, name=key)

            dst.create_dataset("subject_id", data=subject_ids, dtype="int64")

        os.replace(tmp_path, dst_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _update_all_h5(src_path: Path, dst_path: Path, train_subject_ids: np.ndarray, val_subject_ids: np.ndarray) -> None:
    merged_subject_ids = np.concatenate([train_subject_ids, val_subject_ids], axis=0)

    with h5py.File(src_path, "r") as f:
        if len(f["X"]) != len(merged_subject_ids):
            raise ValueError(
                f"{src_path} sample count mismatch: h5={len(f['X'])} merged={len(merged_subject_ids)}"
            )

    _copy_h5_with_subject_id(src_path, dst_path, merged_subject_ids)


def _copy_support_files(src_dir: Path, dst_dir: Path) -> None:
    for name in ("dataset_info.json", "dataset_info_fixed.json", "folds_info.json"):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)

    for fold_dir in sorted(src_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        target_fold = dst_dir / fold_dir.name
        target_fold.mkdir(parents=True, exist_ok=True)
        for npy_file in fold_dir.glob("*.npy"):
            shutil.copy2(npy_file, target_fold / npy_file.name)


def _describe_subjects(subject_ids: np.ndarray) -> str:
    unique_subjects, counts = np.unique(subject_ids, return_counts=True)
    pairs = ", ".join(
        f"{int(subject)}:{int(count)}" for subject, count in zip(unique_subjects, counts)
    )
    return f"{len(unique_subjects)} subjects [{pairs}]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach subject_id metadata to SEED HDF5 splits using teacher-provided txt indices."
    )
    parser.add_argument("--data_dir", default="data/SEED", help="Directory containing SEED HDF5 and txt files")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional destination directory. Defaults to updating data_dir in place.",
    )
    parser.add_argument("--train_txt", default="train_idx.txt", help="Train txt filename inside data_dir")
    parser.add_argument("--val_txt", default="val_idx.txt", help="Val txt filename inside data_dir")
    parser.add_argument(
        "--skip_all_h5",
        action="store_true",
        help="Do not update all.h5 even if it exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate everything and print the planned writes without modifying files.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir

    train_h5 = data_dir / "train.h5"
    val_h5 = data_dir / "val.h5"
    all_h5 = data_dir / "all.h5"
    train_txt = data_dir / args.train_txt
    val_txt = data_dir / args.val_txt

    for path in (train_h5, val_h5, train_txt, val_txt):
        if not path.exists():
            raise FileNotFoundError(path)

    train_subject_ids, train_labels = _parse_index_txt(train_txt)
    val_subject_ids, val_labels = _parse_index_txt(val_txt)

    _validate_split(train_h5, train_subject_ids, train_labels)
    _validate_split(val_h5, val_subject_ids, val_labels)

    print("[ok] train.h5 aligned with train_idx.txt")
    print(f"     {_describe_subjects(train_subject_ids)}")
    print("[ok] val.h5 aligned with val_idx.txt")
    print(f"     {_describe_subjects(val_subject_ids)}")

    if all_h5.exists() and not args.skip_all_h5:
        with h5py.File(all_h5, "r") as f:
            expected = len(train_subject_ids) + len(val_subject_ids)
            if len(f["X"]) != expected:
                raise ValueError(
                    f"{all_h5} sample count mismatch: h5={len(f['X'])} expected={expected}"
                )
        print("[ok] all.h5 size matches train+val merge")

    if args.dry_run:
        print(f"[dry-run] would write subject_id into: {output_dir / 'train.h5'}")
        print(f"[dry-run] would write subject_id into: {output_dir / 'val.h5'}")
        if all_h5.exists() and not args.skip_all_h5:
            print(f"[dry-run] would write subject_id into: {output_dir / 'all.h5'}")
        if output_dir != data_dir:
            print(f"[dry-run] would copy dataset metadata and fold index files into: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    _copy_h5_with_subject_id(train_h5, output_dir / "train.h5", train_subject_ids)
    _copy_h5_with_subject_id(val_h5, output_dir / "val.h5", val_subject_ids)
    print(f"[write] {output_dir / 'train.h5'}")
    print(f"[write] {output_dir / 'val.h5'}")

    test_h5 = data_dir / "test_x_only.h5"
    if output_dir != data_dir and test_h5.exists():
        shutil.copy2(test_h5, output_dir / "test_x_only.h5")

    if all_h5.exists() and not args.skip_all_h5:
        _update_all_h5(all_h5, output_dir / "all.h5", train_subject_ids, val_subject_ids)
        print(f"[write] {output_dir / 'all.h5'}")

    if output_dir != data_dir:
        _copy_support_files(data_dir, output_dir)
        print(f"[copy] support files -> {output_dir}")

    print("[done] subject_id metadata attached successfully")


if __name__ == "__main__":
    main()

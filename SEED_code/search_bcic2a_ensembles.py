"""Search the best BCIC2A validation ensemble over saved run directories.

This script loads validation probabilities from existing result directories,
enumerates combinations up to a requested size, and reports the strongest
soft-voting ensembles on the shared validation split.

Example:
    python search_bcic2a_ensembles.py --runs \
        BCIC2A_ATCNet_20260517_082809_fold1 \
        BCIC2A_ATCNet_20260517_083230_fold1 \
        BCIC2A_ATCNet_20260516_004839_fold1 \
        --max_size 3
"""

import argparse
import itertools
import json
import os
from datetime import datetime

import torch

from model import MODEL_DICT
from utils import create_dataloaders, load_dataset_info


RESULTS_DIR = "Results"


def build_model(cfg):
    channels, num_classes, window_sec = load_dataset_info(cfg["dataset"])
    time_points = int(window_sec * 200)
    model_cls = MODEL_DICT[cfg["model"]]
    kwargs = dict(chans=channels, num_classes=num_classes, time_point=time_points)
    if cfg["model"] == "ATCNet":
        kwargs["n_windows"] = cfg.get("atc_n_windows", 5)
        kwargs["F1"] = cfg.get("atc_f1", 16)
        kwargs["d_model"] = cfg.get("atc_d_model", 32)
        kwargs["dropout_conv"] = cfg.get("atc_dropout_conv", 0.3)
        kwargs["dropout_attn"] = cfg.get("atc_dropout_attn", 0.3)
        kwargs["attn_drop"] = cfg.get("atc_attn_drop", 0.5)
        kwargs["dropout_tcn"] = cfg.get("atc_dropout_tcn", 0.3)
        kwargs["residual_drop"] = cfg.get("atc_residual_drop", 0.0)
        kwargs["drop_path_prob"] = cfg.get("atc_drop_path_prob", 0.0)
        kwargs["tcn_depth"] = cfg.get("atc_tcn_depth", 2)
    return model_cls(**kwargs)


def load_cfg(run_dir):
    with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def collect_val_probs(run_dir):
    cfg = load_cfg(run_dir)
    _, val_loader, _ = create_dataloaders(
        cfg["dataset"],
        cfg["batch_size"],
        fold=cfg.get("fold"),
        standardize=cfg.get("standardize_inputs", False),
    )

    model = build_model(cfg)
    state = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    val_probs = []
    val_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            val_probs.append(torch.softmax(model(x), dim=1))
            val_labels.append(y)

    return torch.cat(val_probs, dim=0), torch.cat(val_labels, dim=0), cfg


def combo_accuracy(items, combo):
    probs = sum(items[idx]["val_probs"] for idx in combo) / len(combo)
    labels = items[combo[0]]["val_labels"]
    preds = probs.argmax(dim=1)
    return (preds == labels).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description="Search BCIC2A ensemble combinations")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories under Results/",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=3,
        help="Maximum ensemble size to evaluate",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="How many top combinations to print",
    )
    args = parser.parse_args()

    run_dirs = [os.path.join(RESULTS_DIR, run) for run in args.runs]
    items = []
    base_cfg = None

    for run_name, run_dir in zip(args.runs, run_dirs):
        val_probs, val_labels, cfg = collect_val_probs(run_dir)
        if base_cfg is None:
            base_cfg = cfg
        else:
            if cfg["dataset"] != base_cfg["dataset"] or cfg.get("fold") != base_cfg.get("fold"):
                raise ValueError("All runs must share the same dataset and fold for fair ensemble search.")
        items.append({
            "name": run_name,
            "val_probs": val_probs,
            "val_labels": val_labels,
        })

    scored = []
    max_size = min(args.max_size, len(items))
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(range(len(items)), size):
            acc = combo_accuracy(items, combo)
            scored.append({
                "size": size,
                "acc": acc,
                "members": [items[idx]["name"] for idx in combo],
            })

    scored.sort(key=lambda row: row["acc"], reverse=True)
    top_rows = scored[: args.top_k]

    print(f"Evaluated {len(scored)} combinations")
    print("")
    for rank, row in enumerate(top_rows, start=1):
        members = ", ".join(row["members"])
        print(f"{rank:02d}. {row['acc'] * 100:.2f}% | size={row['size']} | {members}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"BCIC2A_ensemble_search_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": base_cfg["dataset"],
                "fold": base_cfg.get("fold"),
                "runs": args.runs,
                "max_size": max_size,
                "top_k": args.top_k,
                "results": top_rows,
            },
            f,
            indent=2,
        )
    print(f"\nSaved ranked combinations to: {out_path}")


if __name__ == "__main__":
    main()

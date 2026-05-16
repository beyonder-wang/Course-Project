"""Generate or recover summary.txt for a supervised run directory."""

import argparse
import json
import os
import re
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import write_summary_txt


EPOCH_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\] \| Train Loss: ([0-9.]+) \| Val Loss: ([0-9.]+) \| Val Acc: ([0-9.]+)"
)
BEST_RE = re.compile(r"Best Val Accuracy: ([0-9.]+) \(epoch (\d+)\)")
HEADER_RE = {
    "dataset": re.compile(r"=== Dataset: ([A-Z0-9_]+) ==="),
    "channels": re.compile(r"Channels: (\d+), Classes: (\d+), Time points: (\d+)"),
    "model": re.compile(r"Model: ([^|]+) \| LR: ([0-9.eE+-]+) \| Epochs: (\d+) \| Batch: (\d+)"),
    "device": re.compile(r"Device: (.+)"),
    "amp": re.compile(r"AMP: (True|False) \| Grad accum: (\d+)"),
}


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_run_log(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    parsed = {}
    for key, pattern in HEADER_RE.items():
        match = pattern.search(text)
        if match:
            parsed[key] = match.groups()

    epochs = [m.groups() for m in EPOCH_RE.finditer(text)]
    best_match = BEST_RE.search(text)

    metrics = None
    if epochs:
        train_losses = [float(item[2]) for item in epochs]
        val_losses = [float(item[3]) for item in epochs]
        val_accuracies = [float(item[4]) for item in epochs]
        best_val = max(val_accuracies)
        best_epoch = val_accuracies.index(best_val) + 1
        if best_match:
            best_val = float(best_match.group(1))
            best_epoch = int(best_match.group(2))
        metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "final_val_accuracy": val_accuracies[-1],
            "best_val_accuracy": best_val,
            "best_epoch": best_epoch,
        }
    return parsed, metrics


def _build_config(args, parsed, config_json):
    if config_json:
        return config_json

    config = {}
    if "dataset" in parsed:
        config["dataset"] = parsed["dataset"][0]
    if "model" in parsed:
        model, lr, epochs, batch = parsed["model"]
        config["model"] = model.strip()
        config["lr"] = float(lr)
        config["epochs"] = int(epochs)
        config["batch_size"] = int(batch)
    if "channels" in parsed:
        channels, num_classes, time_points = parsed["channels"]
        config["channels"] = int(channels)
        config["num_classes"] = int(num_classes)
        config["time_points"] = int(time_points)
    if "device" in parsed:
        config["device"] = parsed["device"][0].strip()
    if "amp" in parsed:
        amp, accum = parsed["amp"]
        config["amp"] = amp == "True"
        config["grad_accum_steps"] = int(accum)
    config.setdefault("weight_decay", "unknown")
    config.setdefault("patience", "unknown")
    config.setdefault("scheduler", "unknown")
    return config


def _write_summary(run_dir, config, metrics, fold_label):
    sections = [
        ("Configuration", [
            f"Dataset: {config.get('dataset', 'unknown')}",
            f"Model: {config.get('model', 'unknown')}",
            f"Fold: {fold_label}",
            f"Channels: {config.get('channels', 'unknown')}",
            f"Classes: {config.get('num_classes', 'unknown')}",
            f"Time points: {config.get('time_points', 'unknown')}",
            f"Device: {config.get('device', 'unknown')}",
            f"AMP: {config.get('amp', False)}",
            f"Grad accum steps: {config.get('grad_accum_steps', 1)}",
            f"Learning rate: {config.get('lr', 'unknown')}",
            f"Weight decay: {config.get('weight_decay', 'unknown')}",
            f"Batch size: {config.get('batch_size', 'unknown')}",
            f"Epochs: {config.get('epochs', 'unknown')}",
            f"Patience: {config.get('patience', 'unknown')}",
            f"Scheduler: {config.get('scheduler', 'unknown')}",
        ]),
        ("Results", [
            f"Best val accuracy: {metrics['best_val_accuracy']:.4f} (epoch {metrics['best_epoch']})",
            f"Final val accuracy: {metrics['final_val_accuracy']:.4f}",
            f"Final train loss: {metrics['train_losses'][-1]:.4f}",
            f"Final val loss: {metrics['val_losses'][-1]:.4f}",
        ]),
        ("Output Files", [
            f"Model: {os.path.join(run_dir, 'model.pt')}",
            f"Predictions: {os.path.join(run_dir, 'predictions.txt')}",
            f"Metrics: {os.path.join(run_dir, 'metrics.json')}",
            f"Config: {os.path.join(run_dir, 'config.json')}",
            f"Log: {os.path.join(run_dir, 'run.log')}",
        ]),
    ]
    return write_summary_txt(run_dir, sections)


def main():
    parser = argparse.ArgumentParser(description="Recover summary.txt for a supervised run")
    parser.add_argument("--run_dir", help="Run directory containing run.log and artifacts")
    parser.add_argument("--log_path", help="Standalone run.log path when run_dir is unavailable")
    parser.add_argument("--summary_out", help="Where to write a standalone summary when using --log_path only")
    parser.add_argument("--fold_label", default="original_split", help="Fold label to show in summary")
    args = parser.parse_args()

    if not args.run_dir and not args.log_path:
        raise ValueError("Provide either --run_dir or --log_path")

    if args.run_dir:
        run_dir = args.run_dir
        config_json = _load_json(os.path.join(run_dir, "config.json"))
        metrics_json = _load_json(os.path.join(run_dir, "metrics.json"))
        log_path = args.log_path or os.path.join(run_dir, "run.log")

        parsed = {}
        log_metrics = None
        if os.path.exists(log_path):
            parsed, log_metrics = _parse_run_log(log_path)

        config = _build_config(args, parsed, config_json)
        metrics = metrics_json if metrics_json is not None else log_metrics
        if metrics is None:
            raise FileNotFoundError(
                f"Could not recover metrics from {log_path} or metrics.json in {run_dir}"
            )

        summary_path = _write_summary(run_dir, config, metrics, args.fold_label)
    else:
        log_path = args.log_path
        parsed, metrics = _parse_run_log(log_path)
        if metrics is None:
            raise FileNotFoundError(f"Could not recover metrics from {log_path}")
        config = _build_config(args, parsed, None)

        summary_path = args.summary_out or os.path.join(
            os.path.dirname(log_path),
            os.path.splitext(os.path.basename(log_path))[0] + "_summary.txt",
        )
        sections = [
            ("Configuration", [
                f"Dataset: {config.get('dataset', 'unknown')}",
                f"Model: {config.get('model', 'unknown')}",
                f"Fold: {args.fold_label}",
                f"Channels: {config.get('channels', 'unknown')}",
                f"Classes: {config.get('num_classes', 'unknown')}",
                f"Time points: {config.get('time_points', 'unknown')}",
                f"Device: {config.get('device', 'unknown')}",
                f"AMP: {config.get('amp', False)}",
                f"Grad accum steps: {config.get('grad_accum_steps', 1)}",
                f"Learning rate: {config.get('lr', 'unknown')}",
                f"Weight decay: {config.get('weight_decay', 'unknown')}",
                f"Batch size: {config.get('batch_size', 'unknown')}",
                f"Epochs: {config.get('epochs', 'unknown')}",
                f"Patience: {config.get('patience', 'unknown')}",
                f"Scheduler: {config.get('scheduler', 'unknown')}",
            ]),
            ("Results", [
                f"Best val accuracy: {metrics['best_val_accuracy']:.4f} (epoch {metrics['best_epoch']})",
                f"Final val accuracy: {metrics['final_val_accuracy']:.4f}",
                f"Final train loss: {metrics['train_losses'][-1]:.4f}",
                f"Final val loss: {metrics['val_losses'][-1]:.4f}",
            ]),
            ("Output Files", [
                f"Log: {log_path}",
            ]),
        ]
        out_dir = os.path.dirname(summary_path) or "."
        out_name = os.path.basename(summary_path)
        tmp_path = write_summary_txt(out_dir, sections)
        os.replace(tmp_path, os.path.join(out_dir, out_name))
        summary_path = os.path.join(out_dir, out_name)

    print(f"Recovered summary at: {summary_path}")


if __name__ == "__main__":
    main()

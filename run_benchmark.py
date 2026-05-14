"""Batch benchmark runner — runs multiple models on a single fold and logs results."""
import sys, os, json, time
import torch

os.makedirs("Results", exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from model import MODEL_DICT
from utils import load_dataset_info, create_dataloaders, resolve_device
from trainer import Trainer

DATASET = "BCIC2A"
EPOCHS = 30
LR = 5e-4
BATCH_SIZE = 32
FOLD = 1
DEVICE = resolve_device("cpu")

models_to_run = sys.argv[1:] if len(sys.argv) > 1 else [
    "SimpleLinear", "SimpleMLP", "EEGNet", "EEGLSTM", "EEGGRU", "EEGMamba",
    "EEGNet_SE", "EEGNet_SimAM",
    "EEGLSTM_KAN", "EEGNet_KAN",
]

channels, num_classes, window_sec = load_dataset_info(DATASET)
time_points = int(window_sec * 200)
print(f"Dataset: {DATASET}, Channels: {channels}, Classes: {num_classes}, Time points: {time_points}")
print(f"Device: {DEVICE}, Epochs: {EPOCHS}, LR: {LR}, Batch: {BATCH_SIZE}, Fold: {FOLD}")
print("=" * 60)

results = {}

for model_name in models_to_run:
    if model_name not in MODEL_DICT:
        print(f"\nSKIP: {model_name} not in MODEL_DICT")
        continue

    print(f"\n{'=' * 60}")
    print(f"Running: {model_name}")
    print(f"{'=' * 60}")

    log_path = f"Results/{DATASET}_{model_name}_fold{FOLD}_log.txt"

    try:
        # Model init
        model_cls = MODEL_DICT[model_name]
        if model_name in ("SimpleLinear", "SimpleMLP"):
            model = model_cls(input_channels=channels, time_points=time_points, num_classes=num_classes)
        elif model_name == "EEGNet":
            model = model_cls(chans=channels, num_classes=num_classes, time_point=time_points)
        else:
            model = model_cls(chans=channels, num_classes=num_classes)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Params: {n_params:,}")

        # Data
        train_loader, val_loader, test_loader = create_dataloaders(DATASET, BATCH_SIZE, fold=FOLD)

        # Train
        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            test_loader=test_loader, lr=LR, epochs=EPOCHS, device=DEVICE,
        )

        start = time.time()
        history = trainer.train()
        elapsed = time.time() - start

        best_acc = max(history['val_accuracies'])
        final_acc = history['val_accuracies'][-1]
        best_epoch = history['val_accuracies'].index(best_acc) + 1

        print(f"\nResults: Best={best_acc:.4f} (epoch {best_epoch}), Final={final_acc:.4f}")
        print(f"Time: {elapsed:.1f}s")

        # Save predictions and model
        run_dir = f"Results/{DATASET}_{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(run_dir, exist_ok=True)
        trainer.save_predictions(run_dir)
        torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump({
                "model": model_name,
                "best_val_accuracy": best_acc,
                "final_val_accuracy": final_acc,
                "best_epoch": best_epoch,
                "params": n_params,
                "time_seconds": elapsed,
            }, f, indent=2)

        results[model_name] = {
            "best_acc": round(best_acc, 4),
            "final_acc": round(final_acc, 4),
            "best_epoch": best_epoch,
            "params": n_params,
            "time_seconds": round(elapsed, 1),
        }

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        results[model_name] = {"error": str(e)}

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name, r in results.items():
    if "error" in r:
        print(f"  {name:25s} ERROR: {r['error']}")
    else:
        print(f"  {name:25s} Best={r['best_acc']:.4f}  Final={r['final_acc']:.4f}  Params={r['params']:>8,}  Time={r['time_seconds']}s")

with open("Results/benchmark_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nFull results saved to Results/benchmark_summary.json")

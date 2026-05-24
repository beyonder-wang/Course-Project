"""
Final experiments: run top models with extended epochs.
Sequential runner — no timeout issues.
"""
import sys, os, json, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("Results", exist_ok=True)

from model import MODEL_DICT
from utils import load_dataset_info, create_dataloaders, resolve_device
from trainer import Trainer

DATASET = "BCIC2A"
BATCH = 32
FOLD = 1
DEVICE = resolve_device("cpu")

channels, num_classes, window_sec = load_dataset_info(DATASET)
time_points = int(window_sec * 200)

time_point_models = {"EEGNet", "EEGNet_SE", "EEGNet_SimAM", "EEGNet_SimAM_SE", "EEGNet_KAN"}

# Configuration experiments: (model, epochs, lr)
experiments = [
    ("EEGNet_50ep", "EEGNet", 50, 5e-4),
    ("EEGNet_SE_50ep", "EEGNet_SE", 50, 5e-4),
    ("EEGNet_lr1e-3", "EEGNet", 30, 1e-3),
    ("EEGNet_lr1e-4", "EEGNet", 30, 1e-4),
    ("EEGNet_KAN_50ep", "EEGNet_KAN", 50, 5e-4),
]

log_file = open(f"Results/{DATASET}_final.log", "w", buffering=1)

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()

for tag, model_name, epochs, lr in experiments:
    log(f"\n{'='*60}")
    log(f"RUN: {tag} | model={model_name} epochs={epochs} lr={lr}")
    log(f"{'='*60}")

    try:
        model_cls = MODEL_DICT[model_name]
        if model_name in time_point_models:
            model = model_cls(chans=channels, num_classes=num_classes, time_point=time_points)
        else:
            model = model_cls(chans=channels, num_classes=num_classes)

        n_params = sum(p.numel() for p in model.parameters())
        log(f"Params: {n_params:,}")

        train_loader, val_loader, test_loader = create_dataloaders(DATASET, BATCH, fold=FOLD)

        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            test_loader=test_loader, lr=lr, epochs=epochs, device=DEVICE,
            patience=10,
        )

        start = time.time()
        history = trainer.train()
        elapsed = time.time() - start

        best_acc = max(history['val_accuracies'])
        final_acc = history['val_accuracies'][-1]
        best_epoch = history['val_accuracies'].index(best_acc) + 1

        log(f"\n>>> {tag}: best={best_acc:.4f} final={final_acc:.4f} epoch={best_epoch} time={elapsed:.0f}s params={n_params:,}")

        with open(f"Results/{DATASET}_{tag}_results.json", "w") as f:
            json.dump({"tag": tag, "model": model_name, "epochs": epochs, "lr": lr,
                       "best_acc": best_acc, "final_acc": final_acc, "best_epoch": best_epoch,
                       "params": n_params, "time_s": elapsed, "val_accuracies": history['val_accuracies']}, f, indent=2)

    except Exception as e:
        log(f"ERROR: {tag}: {e}")
        import traceback
        traceback.print_exc()

log(f"\n{'='*60}")
log("ALL DONE")
log_file.close()
print(f"\nLog: Results/{DATASET}_final.log")

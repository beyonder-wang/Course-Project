"""5-fold cross-validation for EEGNet_SE on BCIC2A."""
import sys, os, json, time, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("Results", exist_ok=True)

from model import MODEL_DICT
from utils import load_dataset_info, create_dataloaders, resolve_device
from trainer import Trainer

DATASET = "BCIC2A"
EPOCHS = 30
LR = 5e-4
BATCH = 32
DEVICE = resolve_device("cpu")
MODEL_NAME = "EEGNet_SE"

channels, num_classes, window_sec = load_dataset_info(DATASET)
time_points = int(window_sec * 200)

log_file = open(f"Results/{DATASET}_{MODEL_NAME}_cv.log", "w", buffering=1)
def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()

all_metrics = []

for fold in range(1, 6):
    log(f"\n{'='*50}")
    log(f"FOLD {fold}/5")
    log(f"{'='*50}")

    try:
        model = MODEL_DICT[MODEL_NAME](chans=channels, num_classes=num_classes, time_point=time_points)
        n_params = sum(p.numel() for p in model.parameters())
        log(f"Params: {n_params:,}")

        train_loader, val_loader, test_loader = create_dataloaders(DATASET, BATCH, fold=fold)

        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                          test_loader=test_loader, lr=LR, epochs=EPOCHS, device=DEVICE)

        start = time.time()
        history = trainer.train()
        elapsed = time.time() - start

        best_acc = max(history['val_accuracies'])
        final_acc = history['val_accuracies'][-1]
        best_epoch = history['val_accuracies'].index(best_acc) + 1

        log(f">>> Fold {fold}: best={best_acc:.4f} final={final_acc:.4f} epoch={best_epoch} time={elapsed:.0f}s")
        all_metrics.append({"fold": fold, "best": best_acc, "final": final_acc, "epoch": best_epoch})

        # Save fold result
        os.makedirs(f"Results/{DATASET}_{MODEL_NAME}_CV/fold_{fold}", exist_ok=True)
        with open(f"Results/{DATASET}_{MODEL_NAME}_CV/fold_{fold}/metrics.json", "w") as f:
            json.dump({"best_val_accuracy": best_acc, "final_val_accuracy": final_acc,
                       "best_epoch": best_epoch, "val_accuracies": history['val_accuracies']}, f, indent=2)

    except Exception as e:
        log(f"ERROR Fold {fold}: {e}")
        import traceback
        traceback.print_exc()
        all_metrics.append({"fold": fold, "error": str(e)})

best_list = [m["best"] for m in all_metrics if "best" in m]
final_list = [m["final"] for m in all_metrics if "final" in m]

log(f"\n{'='*50}")
log("CV RESULTS (5 folds)")
log(f"{'='*50}")
for m in all_metrics:
    if "error" in m:
        log(f"  Fold {m['fold']}: ERROR - {m['error']}")
    else:
        log(f"  Fold {m['fold']}: best={m['best']:.4f} final={m['final']:.4f} (epoch {m['epoch']})")

if best_list:
    log(f"\n  Mean best: {np.mean(best_list):.4f} ± {np.std(best_list):.4f}")
    log(f"  Mean final: {np.mean(final_list):.4f} ± {np.std(final_list):.4f}")

    summary = {"model": MODEL_NAME, "dataset": DATASET, "epochs": EPOCHS, "lr": LR,
               "per_fold": {str(i+1): all_metrics[i] for i in range(len(all_metrics))},
               "mean_best": float(np.mean(best_list)), "std_best": float(np.std(best_list)),
               "mean_final": float(np.mean(final_list)), "std_final": float(np.std(final_list))}
    with open(f"Results/{DATASET}_{MODEL_NAME}_CV/cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

log_file.close()
print(f"\nCV complete. Log: Results/{DATASET}_{MODEL_NAME}_cv.log")

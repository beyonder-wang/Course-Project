"""BCIC2A Experiment Round 2: ShallowConvNet + ATCNet.

Target: 65%+ validation accuracy on BCIC2A.
Methods: ShallowConvNet (Schirrmeister 2017), ATCNet (Altaheri 2023).
"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("Results", exist_ok=True)

from model import MODEL_DICT
from utils import load_dataset_info, create_dataloaders, resolve_device

DATASET = "BCIC2A"
DEVICE = resolve_device("cpu")

channels, num_classes, window_sec = load_dataset_info(DATASET)
time_points = int(window_sec * 200)

log_file = open(f"Results/{DATASET}_exp2.log", "w", buffering=1)

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()

# ── Experiment definitions ──────────────────────────────────────────
experiments = [
    # (name, model_key, epochs, lr, fold)
    ("ShallowConvNet_30ep",    "ShallowConvNet", 30, 5e-4, 1),
    ("ShallowConvNet_30ep_lr1e-3", "ShallowConvNet", 30, 1e-3, 1),
    ("ATCNet_nw3_30ep",       "ATCNet",         30, 5e-4, 1),
    ("ATCNet_nw5_30ep",       "ATCNet",         30, 5e-4, 1),
    ("ATCNet_nw5_30ep_lr1e-3","ATCNet",         30, 1e-3, 1),
    ("ShallowConvNet_50ep",   "ShallowConvNet", 50, 5e-4, 1),
    ("ATCNet_nw5_50ep_lr1e-3","ATCNet",         50, 1e-3, 1),
]

# ATCNet always uses n_windows=5 unless specified
done = set()
best_overall = 0.0

for exp_name, model_key, epochs, lr, fold in experiments:
    if exp_name in done:
        continue

    log(f"\n{'='*60}")
    log(f"EXPERIMENT: {exp_name} | model={model_key} epochs={epochs} lr={lr}")
    log(f"{'='*60}")

    try:
        kwargs = dict(chans=channels, num_classes=num_classes, time_point=time_points)
        if model_key == "ATCNet":
            if "nw3" in exp_name:
                kwargs["n_windows"] = 3
            else:
                kwargs["n_windows"] = 5

        model = MODEL_DICT[model_key](**kwargs)
        n_params = sum(p.numel() for p in model.parameters())
        log(f"Params: {n_params:,}")

        train_loader, val_loader, test_loader = create_dataloaders(
            DATASET, 32, fold=fold)

        from trainer import Trainer
        trainer = Trainer(model=model, train_loader=train_loader,
                          val_loader=val_loader, test_loader=test_loader,
                          lr=lr, epochs=epochs, device=DEVICE)

        start = time.time()
        history = trainer.train()
        elapsed = time.time() - start

        best_acc = max(history['val_accuracies'])
        final_acc = history['val_accuracies'][-1]
        best_epoch = history['val_accuracies'].index(best_acc) + 1

        log(f">>> {exp_name}: best={best_acc:.4f} final={final_acc:.4f} "
            f"epoch={best_epoch} time={elapsed:.0f}s params={n_params:,}")

        with open(f"Results/{DATASET}_{exp_name}_result.json", "w") as f:
            json.dump({
                "exp": exp_name, "model": model_key, "epochs": epochs, "lr": lr,
                "best_acc": best_acc, "final_acc": final_acc,
                "best_epoch": best_epoch, "params": n_params, "time_s": elapsed,
                "val_accuracies": history['val_accuracies'],
            }, f, indent=2)

        if best_acc > best_overall:
            best_overall = best_acc
        done.add(exp_name)

        # Check stop condition
        if best_acc >= 0.65:
            log(f"\n*** TARGET REACHED: {best_acc:.4f} >= 65%! Stopping. ***")
            break

    except Exception as e:
        log(f"ERROR {exp_name}: {e}")
        import traceback
        traceback.print_exc()

log(f"\n{'='*60}")
log(f"ALL DONE. Best overall: {best_overall:.4f}")
log(f"{'='*60}")
log_file.close()
print(f"\nLog: Results/{DATASET}_exp2.log")

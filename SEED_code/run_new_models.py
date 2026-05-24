"""Test FBCNet and EEGTCNet on BCIC2A fold 1."""
import os, json, time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("Results", exist_ok=True)

from utils import load_dataset_info, create_dataloaders, resolve_device
from trainer import Trainer
from model.fbcnet import FBCNet
from model.tcnet import EEGTCNet

DATASET = "BCIC2A"
EPOCHS = 30
LR = 5e-4
BATCH = 32
FOLD = 1
DEVICE = resolve_device("cpu")
channels, num_classes, window_sec = load_dataset_info(DATASET)
time_points = int(window_sec * 200)

# Models to test
models = {"FBCNet": FBCNet, "EEGTCNet": EEGTCNet}

log_file = open(f"Results/{DATASET}_new_models.log", "w", buffering=1)
def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()

for name, cls in models.items():
    log(f"\n{'='*50}")
    log(f"MODEL: {name}")
    log(f"{'='*50}")

    try:
        model = cls(chans=channels, num_classes=num_classes, time_point=time_points)
        n_params = sum(p.numel() for p in model.parameters())
        log(f"Params: {n_params:,}")

        train_loader, val_loader, test_loader = create_dataloaders(DATASET, BATCH, fold=FOLD)
        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                          test_loader=test_loader, lr=LR, epochs=EPOCHS, device=DEVICE)

        start = time.time()
        history = trainer.train()
        elapsed = time.time() - start

        best_acc = max(history['val_accuracies'])
        log(f">>> {name}: best={best_acc:.4f} time={elapsed:.0f}s params={n_params:,}")

        with open(f"Results/{DATASET}_{name}_fold1_result.json", "w") as f:
            json.dump({"model": name, "best_acc": best_acc, "params": n_params, "time_s": elapsed}, f, indent=2)

    except Exception as e:
        log(f"ERROR: {name}: {e}")
        import traceback
        traceback.print_exc()

log_file.close()
print(f"\nDone. Log: Results/{DATASET}_new_models.log")

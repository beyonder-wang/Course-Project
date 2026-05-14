"""
Fast screening runner — uses 15 epochs for quick model comparison.
"""
import sys, os, json, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("Results", exist_ok=True)

from model import MODEL_DICT
from utils import load_dataset_info, create_dataloaders, resolve_device
from trainer import Trainer

DATASET = "BCIC2A"
EPOCHS = 15
LR = 5e-4
BATCH = 32
FOLD = 1
DEVICE = resolve_device("cpu")

channels, num_classes, window_sec = load_dataset_info(DATASET)
time_points = int(window_sec * 200)
time_point_models = {"EEGNet", "EEGNet_SE", "EEGNet_SimAM", "EEGNet_SimAM_SE", "EEGNet_KAN"}
input_dim_models = {"SimpleLinear", "SimpleMLP"}

models = sys.argv[1:] if len(sys.argv) > 1 else [
    "EEGNet", "EEGNet_SE", "EEGNet_SimAM", "EEGNet_SimAM_SE",
    "EEGNet_KAN", "EEGLSTM_KAN"
]

log_file = open(f"Results/{DATASET}_fast_screen.log", "w", buffering=1)

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()

for model_name in models:
    if model_name not in MODEL_DICT:
        log(f"SKIP: {model_name}")
        continue

    log(f"\n{'='*50}")
    log(f"MODEL: {model_name}")
    log(f"{'='*50}")

    try:
        model_cls = MODEL_DICT[model_name]
        if model_name in input_dim_models:
            model = model_cls(input_channels=channels, time_points=time_points, num_classes=num_classes)
        elif model_name in time_point_models:
            model = model_cls(chans=channels, num_classes=num_classes, time_point=time_points)
        else:
            model = model_cls(chans=channels, num_classes=num_classes)

        n_params = sum(p.numel() for p in model.parameters())
        log(f"Params: {n_params:,}")

        train_loader, val_loader, test_loader = create_dataloaders(DATASET, BATCH, fold=FOLD)

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

        log(f"\n>>> {model_name}: best={best_acc:.4f} final={final_acc:.4f} epoch={best_epoch} time={elapsed:.0f}s params={n_params:,}")

        with open(f"Results/{DATASET}_{model_name}_fold{FOLD}_quick.json", "w") as f:
            json.dump({"model": model_name, "best_acc": best_acc, "final_acc": final_acc,
                       "best_epoch": best_epoch, "params": n_params, "time_s": elapsed, "epochs": EPOCHS}, f, indent=2)

    except Exception as e:
        log(f"ERROR: {model_name}: {e}")
        import traceback
        traceback.print_exc()

log_file.close()
print(f"\nDone. Log: Results/{DATASET}_fast_screen.log")

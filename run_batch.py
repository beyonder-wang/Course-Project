"""
Batch experiment runner for BCIC2A.
Runs models SEQUENTIALLY in a single process with no timeout issues.
Usage: python run_batch.py [model1 model2 ...]
"""
import sys, os, json, time, torch
os.makedirs("Results", exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MODEL_DICT
from utils import load_dataset_info, create_dataloaders, resolve_device
from trainer import Trainer

DATASET = "BCIC2A"
EPOCHS = 30
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

results = {}
log_path = f"Results/{DATASET}_batch_run.log"
log_file = open(log_path, "w", buffering=1)

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()

for model_name in models:
    if model_name not in MODEL_DICT:
        log(f"SKIP: {model_name} not in MODEL_DICT")
        continue

    log(f"\n{'='*60}")
    log(f"MODEL: {model_name}")
    log(f"{'='*60}")

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

        log(f"\n>>> RESULT: {model_name} fold={FOLD} best={best_acc:.4f} final={final_acc:.4f} epoch={best_epoch} time={elapsed:.0f}s params={n_params:,}")

        # Save result
        out = f"Results/{DATASET}_{model_name}_fold{FOLD}_results.json"
        with open(out, "w") as f:
            json.dump({"model": model_name, "fold": FOLD, "best_acc": best_acc,
                       "final_acc": final_acc, "best_epoch": best_epoch,
                       "params": n_params, "time_s": elapsed}, f, indent=2)

        results[model_name] = {"best": round(best_acc,4), "final": round(final_acc,4), "params": n_params, "time": round(elapsed,1)}

    except Exception as e:
        log(f"ERROR: {model_name}: {e}")
        import traceback
        traceback.print_exc(file=log_file)
        results[model_name] = {"error": str(e)}

log(f"\n{'='*60}")
log("FINAL SUMMARY:")
log(f"{'='*60}")
for name, r in results.items():
    if "error" in r:
        log(f"  {name:25s} FAILED: {r['error']}")
    else:
        log(f"  {name:25s} best={r['best']:.4f}  final={r['final']:.4f}  {r['params']:>8,} params  {r['time']:>6.0f}s")

log_file.close()
log(f"\nFull log: {log_path}")

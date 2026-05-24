"""Robust single-experiment runner. Saves progress after each epoch."""
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

def run_experiment(tag, model_name, epochs, lr):
    log_path = f"Results/{tag}.log"
    result_path = f"Results/{tag}_result.json"

    log_file = open(log_path, "w", buffering=1)
    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    # Allow resume
    if os.path.exists(result_path):
        data = json.load(open(result_path))
        log(f"Already completed: {tag} best={data['best_acc']:.4f}")
        log_file.close()
        return data

    log(f"Starting: {tag} model={model_name} epochs={epochs} lr={lr}")

    model_cls = MODEL_DICT[model_name]
    time_point_models = {"EEGNet", "EEGNet_SE", "EEGNet_SimAM", "EEGNet_SimAM_SE", "EEGNet_KAN"}

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
    )

    start = time.time()
    history = trainer.train()
    elapsed = time.time() - start

    best_acc = max(history['val_accuracies'])
    final_acc = history['val_accuracies'][-1]
    best_epoch = history['val_accuracies'].index(best_acc) + 1

    log(f"\n>>> {tag}: best={best_acc:.4f} final={final_acc:.4f} epoch={best_epoch} time={elapsed:.0f}s")

    result = {"tag": tag, "model": model_name, "epochs": epochs, "lr": lr,
              "best_acc": best_acc, "final_acc": final_acc, "best_epoch": best_epoch,
              "params": n_params, "time_s": elapsed}

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    log_file.close()
    return result

if __name__ == "__main__":
    tag = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "EEGNet"
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 5e-4
    run_experiment(tag, model, epochs, lr)
    print(f"\nDONE: {tag}")

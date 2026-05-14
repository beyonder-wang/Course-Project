"""Run a single model on a single fold and print results to stdout."""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from model import MODEL_DICT
from utils import load_dataset_info, create_dataloaders, resolve_device
from trainer import Trainer
import os, json, time

model_name = sys.argv[1] if len(sys.argv) > 1 else "EEGNet"
fold = int(sys.argv[2]) if len(sys.argv) > 2 else 1
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30

DATASET = "BCIC2A"
LR = 5e-4
BATCH = 32
DEVICE = resolve_device("cpu")

channels, num_classes, window_sec = load_dataset_info(DATASET)
time_points = int(window_sec * 200)
print(f"DATASET={DATASET} MODEL={model_name} FOLD={fold} EPOCHS={epochs} LR={LR} BATCH={BATCH}")
print(f"Channels={channels} Classes={num_classes} TimePoints={time_points}")

# Models that need time_point parameter
time_point_models = {"EEGNet", "EEGNet_SE", "EEGNet_SimAM", "EEGNet_SimAM_SE", "EEGNet_KAN"}
# Models that use input_channels
input_channel_models = {"SimpleLinear", "SimpleMLP"}

model_cls = MODEL_DICT[model_name]
if model_name in input_channel_models:
    model = model_cls(input_channels=channels, time_points=time_points, num_classes=num_classes)
elif model_name in time_point_models:
    model = model_cls(chans=channels, num_classes=num_classes, time_point=time_points)
else:
    model = model_cls(chans=channels, num_classes=num_classes)

n_params = sum(p.numel() for p in model.parameters())
print(f"Params={n_params}")

train_loader, val_loader, test_loader = create_dataloaders(DATASET, BATCH, fold=fold)

trainer = Trainer(
    model=model, train_loader=train_loader, val_loader=val_loader,
    test_loader=test_loader, lr=LR, epochs=epochs, device=DEVICE,
)

start = time.time()
history = trainer.train()
elapsed = time.time() - start

best_acc = max(history['val_accuracies'])
final_acc = history['val_accuracies'][-1]
best_epoch = history['val_accuracies'].index(best_acc) + 1

print(f"\nRESULT: model={model_name} fold={fold} best={best_acc:.4f} final={final_acc:.4f} epoch={best_epoch} time={elapsed:.0f}s params={n_params}")

# Save minimal output
out = f"Results/{DATASET}_{model_name}_fold{fold}_results.json"
with open(out, "w") as f:
    json.dump({"model": model_name, "fold": fold, "best_acc": best_acc, "final_acc": final_acc,
               "best_epoch": best_epoch, "params": n_params, "time_s": elapsed}, f, indent=2)
print(f"Saved: {out}")

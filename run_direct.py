"""Run an experiment directly, capturing all output."""
import sys, os, json

sys.stdout = open(os.devnull, 'w')
os.environ['PYTHONUNBUFFERED'] = '1'

# Redirect stdout to both file and terminal
import builtins
original_print = builtins.print

dataset = sys.argv[1] if len(sys.argv) > 1 else "BCIC2A"
model_name = sys.argv[2] if len(sys.argv) > 2 else "EEGNet"

tag = f"{dataset}_{model_name}_direct"
log_path = f"Results/{tag}_run.log"
log_file = open(log_path, 'w', buffering=1)

def my_print(*args, **kwargs):
    kwargs['file'] = log_file
    original_print(*args, **kwargs)
    kwargs['file'] = sys.__stdout__
    original_print(*args, **kwargs)

builtins.print = my_print

print(f"Running: dataset={dataset}, model={model_name}")

from model import MODEL_DICT
from utils import load_dataset_info, create_dataloaders, resolve_device
from trainer import Trainer

channels, num_classes, window_sec = load_dataset_info(dataset)
time_points = int(window_sec * 200)
print(f"Channels: {channels}, Classes: {num_classes}, Time points: {time_points}")
print(f"Model: {model_name}")

device = resolve_device("cpu")

model_cls = MODEL_DICT[model_name]
if model_name in ("SimpleLinear", "SimpleMLP"):
    model = model_cls(input_channels=channels, time_points=time_points, num_classes=num_classes)
elif model_name == "EEGNet":
    model = model_cls(chans=channels, num_classes=num_classes, time_point=time_points)
else:
    model = model_cls(chans=channels, num_classes=num_classes)

print(f"Model params: {sum(p.numel() for p in model.parameters())}")

train_loader, val_loader, test_loader = create_dataloaders(dataset, batch_size=32, fold=1)

trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                  test_loader=test_loader, lr=5e-4, epochs=30, device=device)

history = trainer.train()

best_acc = max(history['val_accuracies'])
print(f"\nBest val accuracy: {best_acc:.4f}")

# Save predictions
run_dir = f"Results/{tag}"
os.makedirs(run_dir, exist_ok=True)
trainer.save_predictions(run_dir)
torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

metrics = {"best_val_accuracy": best_acc, "val_accuracies": history['val_accuracies']}
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nResults saved to {run_dir}/")
print(f"Log saved to {log_path}")

log_file.close()

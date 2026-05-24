import sys
sys.path.insert(0, '.')
import torch
import numpy as np
import os
import csv
import json
from src.models import (EEGNet, EEGNetOld, EEGNetHybrid, TemporalCNN, CNN_LSTM, MultiScaleCNN)
from src.dataset import EEGH5Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


def try_load_model(path, model_classes):
    """Try loading checkpoint into each model class, return first success."""
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        return None, "FAILED", f"Cannot load file: {e}"

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint if isinstance(checkpoint, dict) else {}

    # Remove 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith('module.') else k
        new_state_dict[key] = v

    # Try strict loading first
    for name, model_cls in model_classes:
        try:
            model = model_cls(chans=20, num_classes=2, time_points=200)
            model.load_state_dict(new_state_dict, strict=True)
            return model, name, None
        except Exception:
            pass

    # Try non-strict loading (allow small mismatch)
    for name, model_cls in model_classes:
        try:
            model = model_cls(chans=20, num_classes=2, time_points=200)
            result = model.load_state_dict(new_state_dict, strict=False)
            missing = result.missing_keys
            unexpected = result.unexpected_keys
            # Accept if missing keys are minor (< 30% of total)
            if len(missing) < len(model.state_dict()) * 0.3:
                return model, f"{name}(partial)", f"missing={missing}, unexpected={unexpected}"
        except Exception:
            pass

    return None, "FAILED", "No model architecture matched"


def evaluate_model(model, dataloader, is_hybrid=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                x, y = batch
            else:
                x = batch
                y = None
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_probs.append(probs.numpy())
            if y is not None:
                all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.concatenate(all_probs, axis=0)

    if len(all_labels) > 0:
        all_labels = np.array(all_labels)
        acc = accuracy_score(all_labels, all_preds)
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)
        return acc, bal_acc, f1, cm, all_probs
    else:
        return None, None, None, None, all_probs


def main():
    os.makedirs('reports', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # Load data
    print("Loading training data for normalization stats...")
    train_ds = EEGH5Dataset('train.h5', mode='train')
    norm_stats = train_ds.get_norm_stats()
    train_ds.save_norm_stats('outputs/norm_stats.npz')

    print("Loading validation data...")
    val_ds = EEGH5Dataset('val.h5', mode='val', norm_stats=norm_stats)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    # All model classes to try (order matters - try most specific first)
    model_classes = [
        ('EEGNetHybrid', EEGNetHybrid),
        ('EEGNet', EEGNet),
        ('EEGNetOld', EEGNetOld),
        ('TemporalCNN', TemporalCNN),
        ('CNN_LSTM', CNN_LSTM),
        ('MultiScaleCNN', MultiScaleCNN),
    ]

    # Scan all .pth files
    pth_files = sorted([f for f in os.listdir('.') if f.endswith('.pth')])
    # Also check outputs/
    for f in sorted(os.listdir('outputs')):
        if f.endswith('.pth'):
            pth_files.append(os.path.join('outputs', f))

    results = []
    probs_dict = {}

    print(f"\nFound {len(pth_files)} checkpoint files\n")
    print(f"{'Checkpoint':<40} {'Model':<20} {'Acc':<8} {'BalAcc':<8} {'F1':<8}")
    print("=" * 90)

    for pth in pth_files:
        try:
            model, model_name, load_note = try_load_model(pth, model_classes)
            if model is None:
                results.append({
                    'checkpoint': pth, 'model': 'FAILED', 'loaded': False,
                    'acc': 0, 'bal_acc': 0, 'f1': 0, 'cm': '', 'note': load_note or ''
                })
                print(f"{pth:<40} {'FAILED':<20} {'N/A':<8} {'N/A':<8} {'N/A':<8}  {load_note or ''}")
                continue

            is_hybrid = 'Hybrid' in model_name
            acc, bal_acc, f1, cm, probs = evaluate_model(model, val_loader, is_hybrid)

            results.append({
                'checkpoint': pth, 'model': model_name, 'loaded': True,
                'acc': acc, 'bal_acc': bal_acc, 'f1': f1,
                'cm': str(cm.tolist()) if cm is not None else '',
                'note': load_note or ''
            })
            probs_dict[pth] = probs
            print(f"{pth:<40} {model_name:<20} {acc:<8.4f} {bal_acc:<8.4f} {f1:<8.4f}")

        except Exception as e:
            results.append({
                'checkpoint': pth, 'model': f'ERROR', 'loaded': False,
                'acc': 0, 'bal_acc': 0, 'f1': 0, 'cm': '', 'note': str(e)[:100]
            })
            print(f"{pth:<40} {'ERROR':<20} {str(e)[:60]}")

    # Save CSV
    with open('reports/checkpoint_eval.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['checkpoint', 'model', 'loaded', 'acc', 'bal_acc', 'f1', 'cm', 'note'])
        writer.writeheader()
        writer.writerows(results)

    # Save markdown
    with open('reports/checkpoint_eval.md', 'w', encoding='utf-8') as f:
        f.write("# Checkpoint Evaluation Results\n\n")
        f.write(f"Total checkpoints scanned: {len(pth_files)}\n\n")
        f.write(f"| Checkpoint | Model | Loaded | Acc | Bal Acc | F1 | Note |\n")
        f.write(f"|---|---|---|---|---|---|---|\n")
        for r in results:
            if r['loaded']:
                f.write(f"| {r['checkpoint']} | {r['model']} | ✅ | {r['acc']:.4f} | {r['bal_acc']:.4f} | {r['f1']:.4f} | {r['note']} |\n")
            else:
                f.write(f"| {r['checkpoint']} | {r['model']} | ❌ | N/A | N/A | N/A | {r['note']} |\n")

        # Best model summary
        loaded_results = [r for r in results if r['loaded']]
        if loaded_results:
            best = max(loaded_results, key=lambda r: r['acc'])
            f.write(f"\n## Best Single Model\n\n")
            f.write(f"- **Checkpoint**: {best['checkpoint']}\n")
            f.write(f"- **Model**: {best['model']}\n")
            f.write(f"- **Val Accuracy**: {best['acc']:.4f}\n")
            f.write(f"- **Balanced Accuracy**: {best['bal_acc']:.4f}\n")
            f.write(f"- **Macro F1**: {best['f1']:.4f}\n")

    # Save probabilities for ensemble
    if probs_dict:
        np.savez('outputs/val_probs.npz', **{k.replace('/', '_').replace('\\', '_'): v for k, v in probs_dict.items()})

    # Also save as JSON for easier downstream use
    eval_summary = {
        'results': [{k: v for k, v in r.items() if k != 'cm'} for r in results],
        'probs_keys': list(probs_dict.keys())
    }
    with open('outputs/checkpoint_eval_summary.json', 'w') as f:
        json.dump(eval_summary, f, indent=2, default=str)

    print(f"\n{'=' * 90}")
    print(f"Results saved to reports/checkpoint_eval.csv and reports/checkpoint_eval.md")
    print(f"Val probabilities saved to outputs/val_probs.npz")

    # Print summary
    loaded = [r for r in results if r['loaded']]
    failed = [r for r in results if not r['loaded']]
    print(f"\nSummary: {len(loaded)} loaded, {len(failed)} failed")
    if loaded:
        best = max(loaded, key=lambda r: r['acc'])
        print(f"Best single model: {best['checkpoint']} ({best['model']}) -> acc={best['acc']:.4f}, f1={best['f1']:.4f}")


if __name__ == '__main__':
    main()

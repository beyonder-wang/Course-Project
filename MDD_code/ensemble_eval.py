"""
Ensemble evaluation script.
Tries various ensemble strategies on val.h5 using all successfully loaded checkpoints.
"""
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
import h5py
import os
import json
import csv
from itertools import combinations
from src.models import EEGNet, EEGNetOld, EEGNetHybrid
from src.dataset import EEGH5Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


def load_model_by_keys(pth_path):
    """Load model by inspecting state_dict key patterns."""
    sd = torch.load(pth_path, map_location='cpu', weights_only=False)
    keys = list(sd.keys())
    first_key = keys[0]

    if first_key.startswith('eegnet.'):
        model = EEGNetHybrid(chans=20, num_classes=2, time_points=200)
        model_name = 'EEGNetHybrid'
    elif first_key.startswith('block1.'):
        model = EEGNetOld(chans=20, num_classes=2, time_points=200)
        model_name = 'EEGNetOld'
    elif first_key.startswith('block1_temporal.'):
        model = EEGNet(chans=20, num_classes=2, time_points=200)
        model_name = 'EEGNet'
    else:
        return None, 'unknown', None

    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, model_name, sd


def get_probs(model, X_tensor):
    """Get softmax probabilities from model."""
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).numpy()
    return probs


def eval_preds(y_true, preds):
    """Compute accuracy, balanced accuracy, macro F1, confusion matrix."""
    acc = accuracy_score(y_true, preds)
    bal = balanced_accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average='macro')
    cm = confusion_matrix(y_true, preds)
    return acc, bal, f1, cm


def eval_probs(y_true, avg_probs):
    """From averaged probabilities, get predictions and metrics."""
    preds = avg_probs.argmax(axis=1)
    return eval_preds(y_true, preds)


def main():
    os.makedirs('reports', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # ---- Load val data (raw, no normalization) ----
    print("Loading validation data...")
    with h5py.File('val.h5', 'r') as f:
        X_raw = f['X'][:].astype(np.float32)
        y_val = f['y'][:].astype(np.int64)
    X_raw_tensor = torch.from_numpy(X_raw)

    # ---- Also load normalized val data ----
    train_ds = EEGH5Dataset('train.h5', mode='train')
    norm_stats = train_ds.get_norm_stats()
    val_ds = EEGH5Dataset('val.h5', mode='val', norm_stats=norm_stats)
    X_norm_tensor = torch.from_numpy(val_ds.X)

    # ---- Load full eval summary ----
    with open('outputs/full_eval_summary.json') as f:
        eval_summary = json.load(f)

    # ---- Collect all model probabilities ----
    print("\nLoading all checkpoints and computing probabilities...")
    model_info = []  # list of {pth, model_name, best_acc, best_mode, probs}

    for item in eval_summary:
        pth = item['pth']
        best_mode = item['best_mode']
        best_acc = item['best_acc']

        model, model_name, _ = load_model_by_keys(pth)
        if model is None:
            print(f"  SKIP {pth}: unknown architecture")
            continue

        if best_mode == 'raw':
            probs = get_probs(model, X_raw_tensor)
        else:
            probs = get_probs(model, X_norm_tensor)

        # Verify accuracy matches
        preds = probs.argmax(axis=1)
        acc_check = (preds == y_val).mean()

        model_info.append({
            'pth': pth,
            'model_name': model_name,
            'best_acc': best_acc,
            'best_mode': best_mode,
            'probs': probs,
        })
        print(f"  {pth:<40} acc={acc_check:.4f} ({best_mode})")

    print(f"\nTotal models loaded: {len(model_info)}")

    # Sort by accuracy descending
    model_info.sort(key=lambda x: x['best_acc'], reverse=True)

    # ---- Ensemble strategies ----
    results = []

    def record(name, probs_avg, description=""):
        acc, bal, f1, cm = eval_probs(y_val, probs_avg)
        results.append({
            'strategy': name,
            'acc': acc,
            'bal_acc': bal,
            'f1': f1,
            'cm': cm.tolist(),
            'description': description,
            'n_models': description.count('+') + 1 if '+' in description else description
        })
        print(f"  {name:<45} acc={acc:.4f}  bal_acc={bal:.4f}  f1={f1:.4f}")

    print("\n" + "=" * 90)
    print("ENSEMBLE EVALUATION")
    print("=" * 90)

    # ---- 1. Single models (for comparison) ----
    print("\n--- Single Models ---")
    for m in model_info:
        record(f"single: {os.path.basename(m['pth'])}", m['probs'],
               os.path.basename(m['pth']))

    # ---- 2. Top-K averages ----
    print("\n--- Top-K Simple Average ---")
    for k in [2, 3, 4, 5, 6, 8, 10, len(model_info)]:
        if k > len(model_info):
            continue
        top_k = model_info[:k]
        avg = np.mean([m['probs'] for m in top_k], axis=0)
        names = '+'.join([os.path.basename(m['pth'])[:15] for m in top_k])
        record(f"top-{k} average", avg, names)

    # ---- 3. Threshold-based (only models with acc >= threshold) ----
    print("\n--- Accuracy Threshold Ensemble ---")
    for thresh in [0.90, 0.91, 0.92, 0.93]:
        filtered = [m for m in model_info if m['best_acc'] >= thresh]
        if len(filtered) >= 2:
            avg = np.mean([m['probs'] for m in filtered], axis=0)
            record(f"acc>={thresh:.2f} ({len(filtered)} models)", avg,
                   '+'.join([os.path.basename(m['pth'])[:15] for m in filtered]))

    # ---- 4. Seed ensemble (ensemble_model_seed_*) ----
    print("\n--- Seed Ensemble ---")
    seed_models = [m for m in model_info if 'seed' in m['pth'] and 'outputs' not in m['pth']]
    if len(seed_models) >= 2:
        avg = np.mean([m['probs'] for m in seed_models], axis=0)
        record(f"seed ensemble ({len(seed_models)} models)", avg,
               '+'.join([os.path.basename(m['pth']) for m in seed_models]))

    # ---- 5. Fold ensemble (best_model_fold_*) ----
    print("\n--- Fold Ensemble ---")
    fold_models = [m for m in model_info if 'fold' in m['pth']]
    if len(fold_models) >= 2:
        avg = np.mean([m['probs'] for m in fold_models], axis=0)
        record(f"fold ensemble ({len(fold_models)} models)", avg,
               '+'.join([os.path.basename(m['pth']) for m in fold_models]))

    # ---- 6. Weighted soft voting (weight = val_acc) ----
    print("\n--- Weighted Soft Voting ---")
    for k in [3, 5, 8, len(model_info)]:
        if k > len(model_info):
            continue
        top_k = model_info[:k]
        weights = np.array([m['best_acc'] for m in top_k])
        weights = weights / weights.sum()
        weighted_avg = sum(w * m['probs'] for w, m in zip(weights, top_k))
        record(f"weighted top-{k}", weighted_avg,
               '+'.join([f"{os.path.basename(m['pth'])[:12]}*{w:.3f}" for m, w in zip(top_k, weights)]))

    # ---- 7. Weighted by (acc - 0.5) to penalize near-random models ----
    print("\n--- Weighted (acc-0.5) Soft Voting ---")
    for k in [3, 5, 8]:
        if k > len(model_info):
            continue
        top_k = model_info[:k]
        weights = np.array([max(m['best_acc'] - 0.5, 0.01) for m in top_k])
        weights = weights / weights.sum()
        weighted_avg = sum(w * m['probs'] for w, m in zip(weights, top_k))
        record(f"weighted(acc-0.5) top-{k}", weighted_avg,
               '+'.join([f"{os.path.basename(m['pth'])[:12]}*{w:.3f}" for m, w in zip(top_k, weights)]))

    # ---- 8. Seed + Hybrid ensemble ----
    print("\n--- Seed + Hybrid Ensemble ---")
    seed_plus_hybrid = [m for m in model_info if 'seed' in m['pth'] or 'hybrid' in m['pth']]
    if len(seed_plus_hybrid) >= 2:
        avg = np.mean([m['probs'] for m in seed_plus_hybrid], axis=0)
        record(f"seed+hybrid ({len(seed_plus_hybrid)} models)", avg,
               '+'.join([os.path.basename(m['pth']) for m in seed_plus_hybrid]))
        # Weighted version
        weights = np.array([m['best_acc'] for m in seed_plus_hybrid])
        weights = weights / weights.sum()
        w_avg = sum(w * m['probs'] for w, m in zip(weights, seed_plus_hybrid))
        record(f"seed+hybrid weighted ({len(seed_plus_hybrid)} models)", w_avg,
               '+'.join([os.path.basename(m['pth']) for m in seed_plus_hybrid]))

    # ---- 9. Exhaustive top-k combinations from top-6 ----
    print("\n--- Exhaustive Combos (top-6) ---")
    top6 = model_info[:min(6, len(model_info))]
    best_combo_acc = 0
    best_combo_name = ""
    best_combo_probs = None

    for r in range(2, min(len(top6) + 1, 7)):
        for combo in combinations(range(len(top6)), r):
            combo_models = [top6[i] for i in combo]
            avg = np.mean([m['probs'] for m in combo_models], axis=0)
            preds = avg.argmax(axis=1)
            acc = (preds == y_val).mean()
            if acc > best_combo_acc:
                best_combo_acc = acc
                best_combo_name = '+'.join([os.path.basename(m['pth'])[:20] for m in combo_models])
                best_combo_probs = avg

    if best_combo_probs is not None:
        record(f"best exhaustive combo", best_combo_probs, best_combo_name)

    # ---- 10. Threshold tuning on best ensemble ----
    print("\n--- Threshold Tuning (on best strategies) ---")
    # Find best result so far
    best_result = max(results, key=lambda r: r['acc'])
    print(f"  Current best: {best_result['strategy']} acc={best_result['acc']:.4f}")

    # Try threshold tuning on top-3 weighted ensemble probabilities
    top3 = model_info[:3]
    weights = np.array([m['best_acc'] for m in top3])
    weights = weights / weights.sum()
    w_avg_probs = sum(w * m['probs'] for w, m in zip(weights, top3))

    best_thresh = 0.5
    best_thresh_acc = 0
    for thresh in np.arange(0.30, 0.70, 0.01):
        preds_t = (w_avg_probs[:, 1] >= thresh).astype(int)
        acc_t = (preds_t == y_val).mean()
        if acc_t > best_thresh_acc:
            best_thresh_acc = acc_t
            best_thresh = thresh

    preds_tuned = (w_avg_probs[:, 1] >= best_thresh).astype(int)
    acc_t, bal_t, f1_t, cm_t = eval_preds(y_val, preds_tuned)
    results.append({
        'strategy': f'threshold-tuned top-3 weighted (t={best_thresh:.2f})',
        'acc': acc_t, 'bal_acc': bal_t, 'f1': f1_t, 'cm': cm_t.tolist(),
        'description': f'threshold={best_thresh:.2f}', 'n_models': 3
    })
    print(f"  threshold-tuned top-3 (t={best_thresh:.2f}):  acc={acc_t:.4f}  bal={bal_t:.4f}  f1={f1_t:.4f}")

    # ---- SUMMARY ----
    print("\n" + "=" * 90)
    print("FINAL RANKING (sorted by accuracy)")
    print("=" * 90)
    results.sort(key=lambda r: (r['acc'], r['f1']), reverse=True)
    for i, r in enumerate(results[:20]):
        marker = " <<<" if i == 0 else ""
        print(f"  {i+1:2d}. {r['strategy']:<50} acc={r['acc']:.4f}  f1={r['f1']:.4f}{marker}")

    # ---- Determine best strategy ----
    best = results[0]
    print(f"\n*** BEST STRATEGY: {best['strategy']}")
    print(f"    Accuracy:  {best['acc']:.4f}")
    print(f"    Balanced:  {best['bal_acc']:.4f}")
    print(f"    Macro F1:  {best['f1']:.4f}")
    print(f"    CM: {best['cm']}")

    # ---- Save results ----
    # CSV
    with open('reports/ensemble_eval.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['strategy', 'acc', 'bal_acc', 'f1', 'cm', 'description'])
        writer.writeheader()
        for r in results:
            writer.writerow({k: v for k, v in r.items() if k in ['strategy', 'acc', 'bal_acc', 'f1', 'cm', 'description']})

    # Markdown
    with open('reports/ensemble_eval.md', 'w', encoding='utf-8') as f:
        f.write("# Ensemble Evaluation Results\n\n")
        f.write(f"Total strategies evaluated: {len(results)}\n\n")
        f.write("| # | Strategy | Acc | Bal Acc | F1 |\n")
        f.write("|---|---|---|---|---|\n")
        for i, r in enumerate(results):
            marker = " ⭐" if i == 0 else ""
            f.write(f"| {i+1} | {r['strategy']}{marker} | {r['acc']:.4f} | {r['bal_acc']:.4f} | {r['f1']:.4f} |\n")

        f.write(f"\n## Best Strategy\n\n")
        f.write(f"- **Strategy**: {best['strategy']}\n")
        f.write(f"- **Accuracy**: {best['acc']:.4f}\n")
        f.write(f"- **Balanced Accuracy**: {best['bal_acc']:.4f}\n")
        f.write(f"- **Macro F1**: {best['f1']:.4f}\n")
        f.write(f"- **Confusion Matrix**: {best['cm']}\n")

    # ---- Save best ensemble config ----
    # Determine which models are used in the best strategy
    best_config = {
        'strategy': best['strategy'],
        'acc': best['acc'],
        'bal_acc': best['bal_acc'],
        'f1': best['f1'],
        'cm': best['cm'],
    }

    # If best is single model
    if best['strategy'].startswith('single:'):
        pth_name = best['strategy'].replace('single: ', '')
        matching = [m for m in model_info if os.path.basename(m['pth']) == pth_name]
        if matching:
            best_config['type'] = 'single'
            best_config['models'] = [{'pth': matching[0]['pth'], 'mode': matching[0]['best_mode'], 'weight': 1.0}]
    else:
        # For ensemble strategies, determine which models and weights to use
        # Use top-k weighted approach with the models that produced best result
        # We need to figure out which models are in the best strategy

        # Default: use top-3 weighted as it's likely among the best
        # Check if best is threshold-tuned
        if 'threshold' in best['strategy']:
            top3 = model_info[:3]
            weights_arr = np.array([m['best_acc'] for m in top3])
            weights_arr = weights_arr / weights_arr.sum()
            best_config['type'] = 'threshold_ensemble'
            best_config['threshold'] = float(best_thresh)
            best_config['models'] = [
                {'pth': m['pth'], 'mode': m['best_mode'], 'weight': float(w)}
                for m, w in zip(top3, weights_arr)
            ]
        elif 'exhaustive' in best['strategy']:
            # Parse the best combo
            best_config['type'] = 'ensemble_average'
            best_config['description'] = best.get('description', '')
            # Use model_info matching
            combo_names = best.get('description', '').split('+')
            matched = []
            for m in model_info:
                bn = os.path.basename(m['pth'])[:20]
                if bn in combo_names:
                    matched.append({'pth': m['pth'], 'mode': m['best_mode'], 'weight': 1.0 / len(combo_names)})
            if matched:
                best_config['models'] = matched
            else:
                # fallback: just store description
                best_config['models'] = [{'pth': m['pth'], 'mode': m['best_mode']} for m in model_info[:3]]
        elif 'weighted top-' in best['strategy']:
            k = int(best['strategy'].split('top-')[1])
            top_k = model_info[:k]
            weights_arr = np.array([m['best_acc'] for m in top_k])
            weights_arr = weights_arr / weights_arr.sum()
            best_config['type'] = 'weighted_ensemble'
            best_config['models'] = [
                {'pth': m['pth'], 'mode': m['best_mode'], 'weight': float(w)}
                for m, w in zip(top_k, weights_arr)
            ]
        elif 'top-' in best['strategy'] and 'average' in best['strategy']:
            k_str = best['strategy'].split('top-')[1].split(' ')[0]
            k = int(k_str)
            top_k = model_info[:k]
            best_config['type'] = 'simple_average'
            best_config['models'] = [
                {'pth': m['pth'], 'mode': m['best_mode'], 'weight': 1.0 / k}
                for m in top_k
            ]
        elif 'seed+hybrid' in best['strategy']:
            sh_models = [m for m in model_info if 'seed' in m['pth'] or 'hybrid' in m['pth']]
            if 'weighted' in best['strategy']:
                weights_arr = np.array([m['best_acc'] for m in sh_models])
                weights_arr = weights_arr / weights_arr.sum()
                best_config['type'] = 'weighted_ensemble'
                best_config['models'] = [
                    {'pth': m['pth'], 'mode': m['best_mode'], 'weight': float(w)}
                    for m, w in zip(sh_models, weights_arr)
                ]
            else:
                best_config['type'] = 'simple_average'
                best_config['models'] = [
                    {'pth': m['pth'], 'mode': m['best_mode'], 'weight': 1.0 / len(sh_models)}
                    for m in sh_models
                ]
        else:
            # Fallback: store top model info
            best_config['type'] = 'unknown_ensemble'
            best_config['models'] = [{'pth': m['pth'], 'mode': m['best_mode']} for m in model_info[:3]]

    with open('outputs/best_ensemble_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)

    print(f"\nResults saved to reports/ensemble_eval.csv, reports/ensemble_eval.md")
    print(f"Best config saved to outputs/best_ensemble_config.json")


if __name__ == '__main__':
    main()

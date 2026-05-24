"""
predict.py - Generate final MDD.txt predictions using best ensemble config.
"""
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
import h5py
import os
import json
import csv
from src.models import EEGNet, EEGNetOld, EEGNetHybrid


def load_model_by_keys(pth_path):
    """Load model by inspecting state_dict key patterns."""
    sd = torch.load(pth_path, map_location='cpu', weights_only=False)
    keys = list(sd.keys())
    first_key = keys[0]

    if first_key.startswith('eegnet.'):
        model = EEGNetHybrid(chans=20, num_classes=2, time_points=200)
    elif first_key.startswith('block1.'):
        model = EEGNetOld(chans=20, num_classes=2, time_points=200)
    elif first_key.startswith('block1_temporal.'):
        model = EEGNet(chans=20, num_classes=2, time_points=200)
    else:
        raise ValueError(f"Unknown model architecture for {pth_path}, first_key={first_key}")

    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def main():
    os.makedirs('outputs', exist_ok=True)

    # Load best ensemble config
    config_path = 'outputs/best_ensemble_config.json'
    with open(config_path) as f:
        config = json.load(f)

    print(f"Strategy: {config['strategy']}")
    print(f"Val Accuracy: {config['acc']:.4f}")
    print(f"Type: {config['type']}")

    # Load test data (raw)
    print("\nLoading test data...")
    with h5py.File('test_x_only.h5', 'r') as f:
        X_test = f['X'][:].astype(np.float32)
    print(f"Test shape: {X_test.shape}")
    X_test_tensor = torch.from_numpy(X_test)

    # Also prepare normalized test data if needed
    from src.dataset import EEGH5Dataset
    norm_stats = EEGH5Dataset.load_norm_stats('outputs/norm_stats.npz')
    X_test_normed = (X_test - norm_stats['mean']) / norm_stats['std']
    X_test_normed_tensor = torch.from_numpy(X_test_normed)

    # Load models and compute weighted probabilities
    models_config = config['models']
    print(f"\nLoading {len(models_config)} models...")

    weighted_probs = np.zeros((len(X_test), 2), dtype=np.float64)
    total_weight = 0.0

    for mc in models_config:
        pth = mc['pth']
        mode = mc['mode']
        weight = mc.get('weight', 1.0 / len(models_config))

        print(f"  Loading {pth} (mode={mode}, weight={weight:.4f})...")
        model = load_model_by_keys(pth)

        # Choose input based on mode
        if mode == 'normed':
            input_tensor = X_test_normed_tensor
        else:
            input_tensor = X_test_tensor

        # Get predictions in batches
        batch_size = 128
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(input_tensor), batch_size):
                batch = input_tensor[i:i+batch_size]
                logits = model(batch)
                probs = torch.softmax(logits, dim=1).numpy()
                all_probs.append(probs)

        probs = np.concatenate(all_probs, axis=0)
        weighted_probs += weight * probs
        total_weight += weight
        print(f"    Done. Prob mean: class0={probs[:, 0].mean():.4f}, class1={probs[:, 1].mean():.4f}")

    # Normalize weights
    weighted_probs /= total_weight

    # Apply threshold
    threshold = config.get('threshold', 0.5)
    print(f"\nApplying threshold: {threshold:.4f}")

    if config['type'] == 'threshold_ensemble':
        predictions = (weighted_probs[:, 1] >= threshold).astype(int)
    else:
        predictions = weighted_probs.argmax(axis=1)

    # Statistics
    n_class0 = (predictions == 0).sum()
    n_class1 = (predictions == 1).sum()
    print(f"\nPrediction distribution:")
    print(f"  Class 0 (Healthy): {n_class0} ({n_class0/len(predictions)*100:.1f}%)")
    print(f"  Class 1 (MDD):     {n_class1} ({n_class1/len(predictions)*100:.1f}%)")

    # Save MDD.txt
    output_path = 'MDD.txt'
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"\nSaved predictions to {output_path} ({len(predictions)} lines)")

    # Save probabilities CSV
    prob_csv_path = 'outputs/test_probabilities.csv'
    with open(prob_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_index', 'prob_0', 'prob_1', 'pred'])
        for i in range(len(predictions)):
            writer.writerow([i, f"{weighted_probs[i, 0]:.6f}", f"{weighted_probs[i, 1]:.6f}", predictions[i]])
    print(f"Saved probabilities to {prob_csv_path}")

    # Also try without threshold tuning (simple argmax) for comparison
    preds_simple = weighted_probs.argmax(axis=1)
    n0_s = (preds_simple == 0).sum()
    n1_s = (preds_simple == 1).sum()
    agree = (preds_simple == predictions).sum()
    print(f"\nComparison: argmax predictions: class0={n0_s}, class1={n1_s}")
    print(f"Agreement with threshold predictions: {agree}/{len(predictions)} ({agree/len(predictions)*100:.1f}%)")


if __name__ == '__main__':
    main()

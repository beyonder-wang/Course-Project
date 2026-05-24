import sys
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import csv
import time
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from src.models import get_model
from src.dataset import EEGH5Dataset
from src.utils import set_seed, get_device


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)
    n = len(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / n, acc, bal_acc, f1, all_preds, all_labels, all_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='eegnet', choices=['eegnet', 'temporalcnn', 'cnn_lstm', 'multiscale'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Training {args.model} | seed={args.seed} | lr={args.lr} | wd={args.weight_decay} | aug={args.augment} | device={device}")

    train_ds = EEGH5Dataset('train.h5', mode='train', augment=args.augment)
    norm_stats = train_ds.get_norm_stats()
    train_ds.save_norm_stats(os.path.join(args.output_dir, 'norm_stats.npz'))

    val_ds = EEGH5Dataset('val.h5', mode='val', norm_stats=norm_stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = get_model(args.model, chans=20, num_classes=2, time_points=200, dropout_rate=args.dropout)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0
    save_name = f"{args.model}_seed{args.seed}"

    log_path = os.path.join(args.output_dir, f'{save_name}_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_bal_acc', 'val_f1', 'lr'])

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_bal_acc, val_f1, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'{save_name}_best.pth'))
            improved = " *"
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
                  f"best={best_val_acc:.4f}{improved}")

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f'{train_loss:.6f}', f'{train_acc:.4f}', f'{val_loss:.6f}',
                           f'{val_acc:.4f}', f'{val_bal_acc:.4f}', f'{val_f1:.4f}', f'{lr:.8f}'])

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best val acc: {best_val_acc:.4f} | Best val F1: {best_val_f1:.4f}")
    print(f"Model saved: {args.output_dir}/{save_name}_best.pth")

    # Final evaluation with best model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, f'{save_name}_best.pth'), map_location=device, weights_only=True))
    _, final_acc, final_bal_acc, final_f1, final_preds, final_labels, _ = evaluate(model, val_loader, criterion, device)
    print(f"\nFinal eval: acc={final_acc:.4f} bal_acc={final_bal_acc:.4f} f1={final_f1:.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(final_labels, final_preds)}")
    print(classification_report(final_labels, final_preds, target_names=['Healthy', 'MDD']))


if __name__ == '__main__':
    main()

import copy
import os
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, lr, epochs,
                 optimizer=None, patience=0, device="cpu", use_amp=False,
                 grad_accum_steps=1, scheduler=None, label_smoothing=0.0,
                 batch_transform=None, mixup_alpha=0.0):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.scheduler = scheduler
        self.batch_transform = batch_transform
        self.mixup_alpha = float(max(0.0, mixup_alpha))

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            model.parameters(), lr=lr
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_state = None
        self.best_epoch = 0

    def _mixup(self, data, label):
        if self.mixup_alpha <= 0 or data.size(0) < 2:
            return data, label, label, 1.0
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        index = torch.randperm(data.size(0), device=data.device)
        mixed = lam * data + (1.0 - lam) * data[index]
        return mixed, label, label[index], lam

    def train(self):
        best_acc = 0.0
        no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss_sum = 0.0
            train_num = 0
            pending_steps = 0
            self.optimizer.zero_grad(set_to_none=True)

            for data, label in self.train_loader:
                data = data.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                if self.batch_transform is not None:
                    data = self.batch_transform(data)
                data, label_a, label_b, lam = self._mixup(data, label)

                autocast_ctx = torch.cuda.amp.autocast if self.use_amp else nullcontext
                with autocast_ctx():
                    output = self.model(data)
                    if self.mixup_alpha > 0 and data.size(0) > 1:
                        loss = lam * self.criterion(output, label_a)
                        loss = loss + (1.0 - lam) * self.criterion(output, label_b)
                    else:
                        loss = self.criterion(output, label)

                raw_loss = loss.detach().item()
                scaled_loss = loss / self.grad_accum_steps
                if self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                pending_steps += 1
                if pending_steps >= self.grad_accum_steps:
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    pending_steps = 0

                n = label.size(0)
                train_loss_sum += raw_loss * n
                train_num += n

            if pending_steps > 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            epoch_train_loss = train_loss_sum / train_num
            self.train_losses.append(epoch_train_loss)

            self.model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_num = 0

            with torch.no_grad():
                for val_data, val_label in self.val_loader:
                    val_data = val_data.to(self.device, non_blocking=True)
                    val_label = val_label.to(self.device, non_blocking=True)

                    autocast_ctx = torch.cuda.amp.autocast if self.use_amp else nullcontext
                    with autocast_ctx():
                        val_output = self.model(val_data)
                        val_loss = self.criterion(val_output, val_label)

                    n = val_label.size(0)
                    val_loss_sum += val_loss.item() * n
                    val_num += n

                    val_pred = torch.argmax(val_output, dim=1)
                    val_correct += (val_pred == val_label).sum().item()

            epoch_val_loss = val_loss_sum / val_num
            epoch_val_acc = val_correct / val_num

            self.val_losses.append(epoch_val_loss)
            self.val_accuracies.append(epoch_val_acc)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(epoch_val_loss)
                else:
                    self.scheduler.step()

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1

            marker = " *" if no_improve == 0 and epoch > 0 else ""
            print(
                f"Epoch [{epoch+1:02d}/{self.epochs}] | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | "
                f"Val Acc: {epoch_val_acc:.4f}{marker}"
            )

            if self.patience > 0 and no_improve >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={self.patience})")
                break

        print("\n" + "-" * 40)
        print(f"Best Val Accuracy: {best_acc:.4f} (epoch {self.best_epoch})")

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "best_val_accuracy": best_acc,
            "best_epoch": self.best_epoch,
        }

    def save_predictions(self, output_dir):
        self.model.eval()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "predictions.txt")

        all_test_labels = []
        with torch.no_grad():
            for test_data in self.test_loader:
                test_data = test_data.to(self.device, non_blocking=True)
                autocast_ctx = torch.cuda.amp.autocast if self.use_amp else nullcontext
                with autocast_ctx():
                    test_output = self.model(test_data)
                test_pred = torch.argmax(test_output, dim=1)
                all_test_labels.extend(test_pred.cpu().tolist())

        with open(output_path, "w", encoding="utf-8") as f:
            for label in all_test_labels:
                f.write(f"{int(label)}\n")

        print(f"Saved {len(all_test_labels)} labels to: {output_path}")
        return output_path

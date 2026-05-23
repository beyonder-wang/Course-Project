import copy
import os
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, lr, epochs,
                 optimizer=None, patience=0, device="cpu", use_amp=False,
                 grad_accum_steps=1, scheduler=None, label_smoothing=0.0,
                 batch_transform=None, mixup_alpha=0.0, emotion_head=None,
                 emotion_dl_alpha=0.0, emotion_aux_weight=1.0,
                 domain_head=None, domain_adv_weight=0.0,
                 domain_target_key=None, grad_clip_norm=0.0,
                 warmup_epochs=0,
                 class_interp_prob=0.0,
                 class_interp_alpha=0.0,
                 prototype_interp_prob=0.0,
                 prototype_interp_alpha=0.0,
                 domain_adv_warmup_epochs=0,
                 domain_adv_start_epoch=0,
                 domain_adv_grl_schedule="none",
                 domain_adv_grl_max=None):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.emotion_head = emotion_head.to(self.device) if emotion_head is not None else None
        self.domain_head = domain_head.to(self.device) if domain_head is not None else None
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.grad_clip_norm = float(max(0.0, grad_clip_norm))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.scheduler = scheduler
        self.batch_transform = batch_transform
        self.mixup_alpha = float(max(0.0, mixup_alpha))
        self.class_interp_prob = float(max(0.0, class_interp_prob))
        self.class_interp_alpha = float(max(0.0, class_interp_alpha))
        self.prototype_interp_prob = float(max(0.0, prototype_interp_prob))
        self.prototype_interp_alpha = float(max(0.0, prototype_interp_alpha))
        self.emotion_dl_alpha = float(max(0.0, emotion_dl_alpha))
        self.emotion_aux_weight = float(max(0.0, emotion_aux_weight))
        self.domain_adv_weight = float(max(0.0, domain_adv_weight))
        self.domain_target_key = domain_target_key
        self.domain_adv_warmup_epochs = max(0, int(domain_adv_warmup_epochs))
        self.domain_adv_start_epoch = max(0, int(domain_adv_start_epoch))
        self.domain_adv_grl_schedule = str(domain_adv_grl_schedule or "none").lower()
        if self.domain_adv_grl_schedule not in {"none", "dann"}:
            raise ValueError(f"Unsupported domain_adv_grl_schedule: {self.domain_adv_grl_schedule!r}")
        if domain_adv_grl_max is None:
            domain_adv_grl_max = getattr(self.domain_head, "grl_lambda", 0.0)
        self.domain_adv_grl_max = float(max(0.0, domain_adv_grl_max))

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            model.parameters(), lr=lr
        )
        self.scaler = self._make_grad_scaler()

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_state = None
        self.best_epoch = 0

    def _make_grad_scaler(self):
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler("cuda", enabled=self.use_amp)
        return torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda")
        return torch.cuda.amp.autocast()

    def _soft_target_cross_entropy(self, logits, target_probs):
        log_probs = F.log_softmax(logits, dim=1)
        return -(target_probs * log_probs).sum(dim=1).mean()

    def _one_hot(self, labels, num_classes, dtype=torch.float32):
        return F.one_hot(labels, num_classes=num_classes).to(dtype=dtype)

    def _unpack_model_output(self, output):
        if isinstance(output, dict):
            logits = output["logits"]
            features = output.get("features")
            aux_loss = output.get("aux_loss")
            return logits, features, aux_loss
        return output, None, None

    def _unpack_supervised_batch(self, batch):
        if not isinstance(batch, (list, tuple)):
            raise TypeError(f"Unexpected batch type: {type(batch)!r}")
        if len(batch) == 2:
            data, label = batch
            metadata = {}
        elif len(batch) == 3:
            data, label, metadata = batch
        else:
            raise ValueError(f"Expected 2 or 3 batch items, got {len(batch)}")
        if metadata is None:
            metadata = {}
        return data, label, metadata

    def _extract_domain_target(self, metadata):
        if (
            self.domain_head is None
            or self.domain_adv_weight <= 0
            or not self.domain_target_key
            or not isinstance(metadata, dict)
        ):
            return None

        target = metadata.get(self.domain_target_key)
        if target is None:
            return None
        return target.to(self.device, non_blocking=True).long()

    def _mixup(self, data, label):
        if self.mixup_alpha <= 0 or data.size(0) < 2:
            return data, label, label, 1.0, None
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        index = torch.randperm(data.size(0), device=data.device)
        mixed = lam * data + (1.0 - lam) * data[index]
        return mixed, label, label[index], lam, index

    def _class_interpolate(self, data, label, metadata):
        if self.class_interp_prob <= 0 or self.class_interp_alpha <= 0 or data.size(0) < 2:
            return data

        mixed = data.clone()
        subject_ids = None
        if isinstance(metadata, dict):
            subject_ids = metadata.get("subject_id")
            if subject_ids is not None:
                subject_ids = subject_ids.to(data.device)

        for idx in range(data.size(0)):
            if np.random.rand() > self.class_interp_prob:
                continue

            same_class = torch.nonzero(label == label[idx], as_tuple=False).flatten()
            same_class = same_class[same_class != idx]
            if same_class.numel() == 0:
                continue

            if subject_ids is not None:
                cross_subject = same_class[subject_ids[same_class] != subject_ids[idx]]
                if cross_subject.numel() > 0:
                    same_class = cross_subject

            partner_pos = torch.randint(same_class.numel(), (1,), device=data.device)
            partner_idx = int(same_class[partner_pos].item())
            lam = float(np.random.beta(self.class_interp_alpha, self.class_interp_alpha))
            lam = min(0.8, max(0.2, lam))
            mixed[idx] = lam * data[idx] + (1.0 - lam) * data[partner_idx]
        return mixed

    def _prototype_interpolate(self, data, label, metadata):
        if self.prototype_interp_prob <= 0 or self.prototype_interp_alpha <= 0 or data.size(0) < 2:
            return data

        mixed = data.clone()
        subject_ids = None
        if isinstance(metadata, dict):
            subject_ids = metadata.get("subject_id")
            if subject_ids is not None:
                subject_ids = subject_ids.to(data.device)

        unique_labels = torch.unique(label)
        class_prototypes = {}
        class_subject_prototypes = {}
        for cls in unique_labels.tolist():
            cls_mask = label == cls
            class_prototypes[int(cls)] = data[cls_mask].mean(dim=0)
            if subject_ids is None:
                continue

            subject_proto = {}
            cls_subjects = torch.unique(subject_ids[cls_mask])
            for subject in cls_subjects.tolist():
                mask = cls_mask & (subject_ids == subject)
                subject_proto[int(subject)] = data[mask].mean(dim=0)
            class_subject_prototypes[int(cls)] = subject_proto

        for idx in range(data.size(0)):
            if np.random.rand() > self.prototype_interp_prob:
                continue

            cls = int(label[idx].item())
            prototype = class_prototypes.get(cls)
            if prototype is None:
                continue

            if subject_ids is not None:
                current_subject = int(subject_ids[idx].item())
                other_subject_protos = [
                    proto for subject, proto in class_subject_prototypes.get(cls, {}).items()
                    if subject != current_subject
                ]
                if other_subject_protos:
                    prototype = torch.stack(other_subject_protos, dim=0).mean(dim=0)

            lam = float(np.random.beta(self.prototype_interp_alpha, self.prototype_interp_alpha))
            lam = min(0.8, max(0.2, lam))
            mixed[idx] = lam * data[idx] + (1.0 - lam) * prototype
        return mixed

    def _get_effective_adv_weight(self, epoch):
        """Return effective adversarial weight with optional delayed start/warmup."""
        if self.domain_adv_weight <= 0 or epoch < self.domain_adv_start_epoch:
            return 0.0
        if self.domain_adv_warmup_epochs <= 0:
            return self.domain_adv_weight
        active_epoch = epoch - self.domain_adv_start_epoch
        warmup_frac = min(1.0, (active_epoch + 1) / max(1, self.domain_adv_warmup_epochs))
        return warmup_frac * self.domain_adv_weight

    def _get_effective_grl_lambda(self, epoch):
        """Return effective GRL lambda with optional delayed DANN schedule."""
        if self.domain_head is None or self.domain_adv_weight <= 0 or epoch < self.domain_adv_start_epoch:
            return 0.0

        if self.domain_adv_grl_schedule == "dann":
            active_epochs = max(1, self.epochs - self.domain_adv_start_epoch)
            progress = min(1.0, max(0.0, (epoch - self.domain_adv_start_epoch + 1) / active_epochs))
            coeff = 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0
            return self.domain_adv_grl_max * float(coeff)

        return self.domain_adv_grl_max

    def train(self):
        best_acc = 0.0
        no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss_sum = 0.0
            train_num = 0
            pending_steps = 0
            self.optimizer.zero_grad(set_to_none=True)
            effective_adv_weight = self._get_effective_adv_weight(epoch)
            effective_grl_lambda = self._get_effective_grl_lambda(epoch)
            if self.domain_head is not None:
                self.domain_head.grl_lambda = effective_grl_lambda

            for batch in self.train_loader:
                data, label, metadata = self._unpack_supervised_batch(batch)
                data = data.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                domain_target = self._extract_domain_target(metadata)
                data = self._class_interpolate(data, label, metadata)
                data = self._prototype_interpolate(data, label, metadata)
                if self.batch_transform is not None:
                    data = self.batch_transform(data)
                data, label_a, label_b, lam, mix_index = self._mixup(data, label)

                with self._autocast_context():
                    output = self.model(data)
                    logits, features, aux_loss = self._unpack_model_output(output)
                    mixup_enabled = mix_index is not None and data.size(0) > 1
                    if mixup_enabled:
                        hard_target_probs = (
                            lam * self._one_hot(label_a, num_classes=logits.size(1), dtype=logits.dtype)
                            + (1.0 - lam) * self._one_hot(label_b, num_classes=logits.size(1), dtype=logits.dtype)
                        )
                        loss = lam * self.criterion(logits, label_a)
                        loss = loss + (1.0 - lam) * self.criterion(logits, label_b)
                    else:
                        hard_target_probs = self._one_hot(label, num_classes=logits.size(1), dtype=logits.dtype)
                        loss = self.criterion(logits, label)

                    if self.emotion_head is not None and features is not None:
                        emotion_logits = self.emotion_head(features)
                        if self.emotion_dl_alpha > 0:
                            emotion_probs = torch.softmax(emotion_logits, dim=1)
                            target_probs = (1.0 - self.emotion_dl_alpha) * hard_target_probs
                            target_probs = target_probs + self.emotion_dl_alpha * emotion_probs.detach()
                            loss = self._soft_target_cross_entropy(logits, target_probs)

                        if mixup_enabled:
                            emotion_aux_loss = lam * self.criterion(emotion_logits, label_a)
                            emotion_aux_loss = emotion_aux_loss + (1.0 - lam) * self.criterion(
                                emotion_logits, label_b
                            )
                        else:
                            emotion_aux_loss = self.criterion(emotion_logits, label)
                        loss = loss + self.emotion_aux_weight * emotion_aux_loss

                    if aux_loss is not None:
                        loss = loss + aux_loss
                    if (
                        self.domain_head is not None
                        and effective_adv_weight > 0
                        and features is not None
                        and domain_target is not None
                    ):
                        domain_logits = self.domain_head(features)
                        if mix_index is not None and domain_target.size(0) > 1:
                            domain_target_b = domain_target[mix_index]
                            domain_loss = lam * self.criterion(domain_logits, domain_target)
                            domain_loss = domain_loss + (1.0 - lam) * self.criterion(
                                domain_logits, domain_target_b
                            )
                        else:
                            domain_loss = self.criterion(domain_logits, domain_target)
                        loss = loss + effective_adv_weight * domain_loss

                raw_loss = loss.detach().item()
                scaled_loss = loss / self.grad_accum_steps
                if self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                pending_steps += 1
                if pending_steps >= self.grad_accum_steps:
                    if self.grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
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
                if self.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
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
                for batch in self.val_loader:
                    val_data, val_label, _ = self._unpack_supervised_batch(batch)
                    val_data = val_data.to(self.device, non_blocking=True)
                    val_label = val_label.to(self.device, non_blocking=True)

                    with self._autocast_context():
                        val_output = self.model(val_data)
                        val_logits, _, aux_loss = self._unpack_model_output(val_output)
                        val_loss = self.criterion(val_logits, val_label)
                        if aux_loss is not None:
                            val_loss = val_loss + aux_loss

                    n = val_label.size(0)
                    val_loss_sum += val_loss.item() * n
                    val_num += n

                    val_pred = torch.argmax(val_logits, dim=1)
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

            # Linear LR warmup: override scheduler during warmup phase
            if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                lr_scale = min(1.0, (epoch + 1) / self.warmup_epochs)
                warmup_lr = self.lr * lr_scale
                for pg in self.optimizer.param_groups:
                    pg['lr'] = warmup_lr

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
                f"Val Acc: {epoch_val_acc:.4f}"
                f"{' | AdvW: %.3f | GRL: %.3f' % (effective_adv_weight, effective_grl_lambda) if self.domain_head is not None else ''}"
                f"{marker}"
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
                with self._autocast_context():
                    test_output = self.model(test_data)
                    test_logits, _, _ = self._unpack_model_output(test_output)
                test_pred = torch.argmax(test_logits, dim=1)
                all_test_labels.extend(test_pred.cpu().tolist())

        with open(output_path, "w", encoding="utf-8") as f:
            for label in all_test_labels:
                f.write(f"{int(label)}\n")

        print(f"Saved {len(all_test_labels)} labels to: {output_path}")
        return output_path

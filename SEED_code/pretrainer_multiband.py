"""Multi-band SimCLR contrastive pre-training loop.

Handles:
  - Frequency band decomposition inside the encoder
  - Multi-head NT-Xent loss across bands
  - Optional MoE load balancing loss
"""

import torch
import torch.nn as nn

from model.multiband_loss import MultiBandNTXentLoss


class MultiBandPretrainer:
    """Multi-band SimCLR pre-training loop.

    Args:
        model: MultiBandSimCLREncoder or MultiBandMoESimCLREncoder
        train_loader: DataLoader yielding (x,) or (x, source_idx) for Phase 2
        lr: learning rate
        epochs: number of pre-training epochs
        temperature: NT-Xent temperature (default 0.1)
        transform: SimCLRTransform instance (applied BEFORE band decomposition)
        balance_weight: MoE load balancing loss weight (0 = disabled)
        device: torch device string
    """

    def __init__(
        self,
        model,
        train_loader,
        lr,
        epochs,
        temperature=0.1,
        transform=None,
        balance_weight=0.01,
        device="cpu",
    ):
        if transform is None:
            raise ValueError("transform is required (e.g. SimCLRTransform instance)")

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.lr = lr
        self.epochs = epochs
        self.balance_weight = balance_weight

        self.criterion = MultiBandNTXentLoss(temperature=temperature)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.transform = transform

        self.train_losses = []
        self.balance_losses = []

    def train(self):
        """Run multi-band SimCLR pre-training.

        Returns:
            dict with train_losses and balance_losses lists
        """
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_balance = 0.0
            n_batches = 0

            for batch in self.train_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)

                # Apply SimCLR augmentations (before band decomposition)
                x_i, x_j = self.transform(x)

                self.optimizer.zero_grad()

                # Multi-band encoding: get per-band features for both views
                out_i = self.model(x_i)
                out_j = self.model(x_j)

                # Handle MoE balance loss
                bal = torch.tensor(0.0, device=self.device)
                if isinstance(out_i, tuple):
                    z_i, bal_i = out_i
                    z_j, bal_j = out_j
                    bal = (bal_i + bal_j) * 0.5
                else:
                    z_i, z_j = out_i, out_j

                # Multi-head NT-Xent loss
                loss = self.criterion(z_i, z_j)
                total_loss = loss + self.balance_weight * bal
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_balance += bal.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            avg_balance = epoch_balance / n_batches
            self.train_losses.append(avg_loss)
            self.balance_losses.append(avg_balance)

            if avg_balance > 0:
                print(
                    f"Epoch [{epoch + 1:03d}/{self.epochs}] | "
                    f"Pretrain Loss: {avg_loss:.4f} | Balance: {avg_balance:.4f}"
                )
            else:
                print(
                    f"Epoch [{epoch + 1:03d}/{self.epochs}] | "
                    f"Pretrain Loss: {avg_loss:.4f}"
                )

        print(f"\nMulti-band pretraining complete. Final loss: {self.train_losses[-1]:.4f}")
        return {
            "train_losses": self.train_losses,
            "balance_losses": self.balance_losses,
        }

    def save_checkpoint(self, encoder_path):
        """Save encoder state dict (LSTM + optional MoE, excluding projection heads).

        Args:
            encoder_path: path to save encoder.pt
        """
        state = self.model.get_encoder_state_dict()
        torch.save(state, encoder_path)
        print(f"Encoder saved to: {encoder_path}")

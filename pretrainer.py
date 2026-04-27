import torch
from model.augmentations import SimCLRTransform
from model.contrastive_loss import NTXentLoss


class Pretrainer:
    """SimCLR contrastive pre-training loop.

    Args:
        model: SimCLREncoder or MoESimCLREncoder instance
        train_loader: DataLoader yielding (x,) or (x, source_idx) batches
        lr: learning rate
        epochs: number of pre-training epochs
        temperature: NT-Xent temperature (default 0.1)
        transform: SimCLRTransform instance for data augmentation
        is_phase2: if True, loader yields (x, source_idx) and model receives source_idx
        balance_weight: weight for MoE load balancing loss (default 0.01, 0 = disabled)
        device: torch device string (default "cpu")
    """

    def __init__(self, model, train_loader, lr, epochs, temperature=0.1,
                 transform=None, is_phase2=False, balance_weight=0.01, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.lr = lr
        self.epochs = epochs
        self.is_phase2 = is_phase2
        self.balance_weight = balance_weight

        self.criterion = NTXentLoss(temperature=temperature)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.transform = transform

        self.train_losses = []
        self.balance_losses = []

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_balance = 0.0
            n_batches = 0

            for batch in self.train_loader:
                if self.is_phase2:
                    x, source_idx = batch
                    source_idx = source_idx.to(self.device)
                else:
                    x = batch

                x = x.to(self.device)
                x_i, x_j = self.transform(x)

                self.optimizer.zero_grad()

                if self.is_phase2:
                    out_i = self.model(x_i, source_idx)
                    out_j = self.model(x_j, source_idx)
                else:
                    out_i = self.model(x_i)
                    out_j = self.model(x_j)

                bal = torch.tensor(0.0, device=self.device)
                if isinstance(out_i, tuple):
                    z_i, bal_i = out_i
                    z_j, bal_j = out_j
                    bal = (bal_i + bal_j) * 0.5
                else:
                    z_i, z_j = out_i, out_j

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
                print(f"Epoch [{epoch+1:03d}/{self.epochs}] | "
                      f"Pretrain Loss: {avg_loss:.4f} | Balance: {avg_balance:.4f}")
            else:
                print(f"Epoch [{epoch+1:03d}/{self.epochs}] | "
                      f"Pretrain Loss: {avg_loss:.4f}")

        print(f"\nPretraining complete. Final loss: {self.train_losses[-1]:.4f}")
        return {"train_losses": self.train_losses, "balance_losses": self.balance_losses}

    def save_checkpoint(self, encoder_path):
        """Save encoder state dict for downstream fine-tuning."""
        state = self.model.get_encoder_state_dict()
        torch.save(state, encoder_path)
        print(f"Encoder saved to: {encoder_path}")

import os
import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, lr, epochs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr = lr
        self.epochs = epochs

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss_sum = 0.0
            train_num = 0

            for data, label in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                n = label.size(0)
                train_loss_sum += loss.item() * n
                train_num += n

            epoch_train_loss = train_loss_sum / train_num
            self.train_losses.append(epoch_train_loss)

            self.model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_num = 0

            with torch.no_grad():
                for val_data, val_label in self.val_loader:
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

            print(
                f"Epoch [{epoch+1:02d}/{self.epochs}] | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | "
                f"Val Acc: {epoch_val_acc:.4f}"
            )

        print("\n" + "-" * 40)
        print(f"Final Val Accuracy: {self.val_accuracies[-1]:.4f}")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
        }

    def save_predictions(self, data_name, output_dir="Results"):
        self.model.eval()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{data_name}.txt")

        all_test_labels = []
        with torch.no_grad():
            for test_data in self.test_loader:
                test_output = self.model(test_data)
                test_pred = torch.argmax(test_output, dim=1)
                all_test_labels.extend(test_pred.cpu().tolist())

        with open(output_path, "w", encoding="utf-8") as f:
            for label in all_test_labels:
                f.write(f"{int(label)}\n")

        print(f"Saved {len(all_test_labels)} labels to: {output_path}")
        return output_path

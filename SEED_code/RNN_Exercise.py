import torch
import torch.nn as nn


class ExerciseEEGSimpleRNN(nn.Module):
    """
    Complete the 6 blanks for SimpleRNN.
    """

    def __init__(
        self,
        chans=20,
        hidden_dim=64,
        num_layers=2,
        num_classes=3,
        dropout=0.3,
        bidirectional=True,
        grad_clip=1.0,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.grad_clip = grad_clip

        self.rnn = nn.RNN(
            input_size=chans,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="relu",
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self._init_rnn_weights()

        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def _init_rnn_weights(self):
        for name, param in self.rnn.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # Input shape: (B, C, T)
        x = x.transpose(1, 2)

        out, h_n = self.rnn(x)

        if self.bidirectional:
            feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            feat = h_n[-1]

        logits = self.classifier(feat)
        return logits

    def clip_gradients(self):
        return torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)


class ExerciseEEGLSTM(nn.Module):
    """
    Complete the 6 blanks for LSTM.
    """

    def __init__(
        self,
        chans=20,
        hidden_dim=64,
        num_layers=2,
        num_classes=3,
        dropout=0.3,
        bidirectional=True,
        grad_clip=1.0,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.grad_clip = grad_clip

        self.lstm = nn.LSTM(
            input_size=chans,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # Input shape: (B, C, T)
        x = x.transpose(1, 2)

        out, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            feat = h_n[-1]

        logits = self.classifier(feat)
        return logits

    def clip_gradients(self):
        return torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # input shape: (batch, time, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)          # out: (batch, time, hidden)
        last = out[:, -1, :]          # take the final time step
        return self.fc(self.dropout(last)).squeeze(-1)   # (batch,)


class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)         # the LSTM also returns (hidden, cell), which we ignore
        last = out[:, -1, :]
        return self.fc(self.dropout(last)).squeeze(-1)


if __name__ == "__main__":
    # Quick self-test with random data (no training)
    fake_batch = torch.randn(64, 30, 4)   # 64 windows, 30 days, 4 features

    for model in (GRUModel(), LSTMModel()):
        print(model)
        output = model(fake_batch)
        n_params = sum(p.numel() for p in model.parameters())
        print("Output shape:", tuple(output.shape), "| Trainable parameters:", n_params)
        print()
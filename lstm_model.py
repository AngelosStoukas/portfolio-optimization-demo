import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: [batch, sequence_length, features]
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]) # Πρόβλεψη για την επόμενη ημέρα
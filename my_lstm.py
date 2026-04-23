import torch
import torch.nn as nn

class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(StackedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Το κύριο σώμα του LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        
        # Ένα πλήρως συνδεδεμένο επίπεδο για την τελική πρόβλεψη
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Αρχικοποίηση κρυφών καταστάσεων (h0, c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass μέσα από το LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out

# Παράμετροι
# input_size = 5 (5 μετοχές), hidden_size = 64, num_layers = 2
model = StackedLSTM(input_size=5, hidden_size=64, num_layers=2, output_size=5)
print(model)
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_market_data, to_tensor
from lstm_model import LSTMForecaster
from tsfm_model import TSFMForecaster
from base_model import PortfolioOptimizer

# 1. Δεδομένα και Παράμετροι
tickers = ['AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT']
n_assets = len(tickers)
window_size = 30

print("Λήψη δεδομένων από το Yahoo Finance...")
# Παίρνουμε δεδομένα από το 2020 για να έχουμε υλικό εκπαίδευσης
all_data = get_market_data(tickers, start="2020-01-01", end="2024-12-31")
all_tensor = to_tensor(all_data)

# Χωρίζουμε τα δεδομένα: Εκπαίδευση (2020-2022) και Test (2023-2024)
train_limit = len(all_data[all_data.index < "2023-01-01"])
train_tensor = all_tensor[:train_limit]
test_tensor = all_tensor[train_limit:]

# 2. Αρχικοποίηση Μοντέλων
lstm = LSTMForecaster(n_assets, 64)
tsfm = TSFMForecaster() # To Chronos είναι ήδη προ-εκπαιδευμένο
optimizer_net = PortfolioOptimizer(n_assets, n_assets)

# --- ΕΚΠΑΙΔΕΥΣΗ LSTM (Το κλειδί για να ξεχωρίσουν οι γραμμές) ---
print("Έναρξη εκπαίδευσης LSTM στο Training Set (2020-2022)...")
criterion = torch.nn.MSELoss()
optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=0.001)

lstm.train()
for epoch in range(15): # 15 εποχές είναι αρκετές για το demo
    epoch_loss = 0
    for i in range(window_size, len(train_tensor)-1):
        x = train_tensor[i-window_size:i].unsqueeze(0)
        y = train_tensor[i+1]
        
        optimizer_lstm.zero_grad()
        pred = lstm(x)
        loss = criterion(pred, y.unsqueeze(0))
        loss.backward()
        optimizer_lstm.step()
        epoch_loss += loss.item()
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/15 | Loss: {epoch_loss/len(train_tensor):.6f}")

# 3. Backtesting στο Test Set (2023-2024)
results = {'lstm_val': [100.0], 'tsfm_val': [100.0], 'actual': [], 'l_pred': [], 't_pred': []}

print("Έναρξη Backtesting (Out-of-sample)...")
lstm.eval()
with torch.no_grad():
    for t in range(window_size, len(test_tensor) - 1):
        context = test_tensor[t-window_size:t]
        actual = test_tensor[t+1]
        
        # Προβλέψεις (Επίπεδο Α)
        p_l = lstm(context.unsqueeze(0)).flatten()
        p_t = tsfm.predict(context.T).flatten()
        
        # Βάρη Χαρτοφυλακίου (Επίπεδο Β)
        w_l = optimizer_net(p_l.unsqueeze(0))
        w_t = optimizer_net(p_t.unsqueeze(0))
        
        # Πραγματική Απόδοση
        r_l = torch.sum(w_l * actual).item()
        r_t = torch.sum(w_t * actual).item()
        
        results['lstm_val'].append(results['lstm_val'][-1] * (1 + r_l))
        results['tsfm_val'].append(results['tsfm_val'][-1] * (1 + r_t))
        results['actual'].append(actual.numpy())
        results['l_pred'].append(p_l.numpy())
        results['t_pred'].append(p_t.numpy())

# 4. Υπολογισμός MAE
mae_l = np.mean(np.abs(np.array(results['actual']) - np.array(results['l_pred'])))
mae_t = np.mean(np.abs(np.array(results['actual']) - np.array(results['t_pred'])))
print(f"\nΣτατιστικό Σφάλμα (MAE) -> LSTM: {mae_l:.4f} | TSFM: {mae_t:.4f}")

# 5. Οπτικοποίηση
plt.figure(figsize=(12,6))
plt.plot(results['lstm_val'], label='Στρατηγική LSTM (Trained)')
plt.plot(results['tsfm_val'], label='Στρατηγική TSFM (Chronos Zero-shot)')
plt.title("Σύγκριση Απόδοσης Χαρτοφυλακίου (Level B)")
plt.xlabel("Ημέρες Διακράτησης")
plt.ylabel("Αξία Επένδυσης (€)")
plt.legend()
plt.grid(True)
plt.show()
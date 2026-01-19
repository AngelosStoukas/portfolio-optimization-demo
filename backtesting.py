import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_market_data, to_tensor
from lstm_model import LSTMForecaster
from tsfm_model import TSFMForecaster
from base_model import PortfolioOptimizer, sharpe_loss

# 1. Προετοιμασία Δεδομένων
tickers = ['AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT']
data = get_market_data(tickers, start="2023-01-01", end="2024-12-31")
returns_tensor = to_tensor(data)

# 2. Αρχικοποίηση Μοντέλων
n_assets = len(tickers)
window_size = 30
lstm = LSTMForecaster(n_assets, 64)
tsfm = TSFMForecaster()
optimizer = PortfolioOptimizer(n_assets, n_assets)

# Αποθήκες για τα αποτελέσματα
results = {
    'actual_returns': [],
    'lstm_preds': [],
    'tsfm_preds': [],
    'portfolio_val_lstm': [100.0], # αρχικο κεφαλαιο 100€
    'portfolio_val_tsfm': [100.0]
}

print("Έναρξη Backtesting...")

# 3. Rolling Window Loop μεθοδος 
for t in range(window_size, len(returns_tensor) - 1):
    # Παίρνουμε το παρελθόν (context)
    context = returns_tensor[t-window_size:t]
    # Τι πραγματικά συνέβη την επόμενη μέρα
    actual_next_day = returns_tensor[t+1]
    
    # ΕΠΙΠΕΔΟ Α: ΠΡΟΒΛΕΨΗ 
    with torch.no_grad():
        p_lstm = lstm(context.unsqueeze(0)).flatten()
        p_tsfm = tsfm.predict(context.T).flatten()
    
    results['lstm_preds'].append(p_lstm.numpy())
    results['tsfm_preds'].append(p_tsfm.numpy())
    results['actual_returns'].append(actual_next_day.numpy())

    # ΕΠΙΠΕΔΟ Β: ΑΠΟΔΟΣΗ ΧΑΡΤΟΦΥΛΑΚΙΟΥ
    w_lstm = optimizer(p_lstm.unsqueeze(0))
    w_tsfm = optimizer(p_tsfm.unsqueeze(0))
    
    # Υπολογισμός κέρδους/ζημιάς ημέρας
    ret_lstm = torch.sum(w_lstm * actual_next_day).item()
    ret_tsfm = torch.sum(w_tsfm * actual_next_day).item()
    
    # Ενημέρωση αξίας κεφαλαίου
    results['portfolio_val_lstm'].append(results['portfolio_val_lstm'][-1] * (1 + ret_lstm))
    results['portfolio_val_tsfm'].append(results['portfolio_val_tsfm'][-1] * (1 + ret_tsfm))

# 4. Υπολογισμός MAE στατιστικό σφάλμα (Επίπεδο Α)
mae_lstm = np.mean(np.abs(np.array(results['actual_returns']) - np.array(results['lstm_preds'])))
mae_tsfm = np.mean(np.abs(np.array(results['actual_returns']) - np.array(results['tsfm_preds'])))

print(f"\nΣτατιστικό Σφάλμα (MAE) - LSTM: {mae_lstm:.4f}")
print(f"Στατιστικό Σφάλμα (MAE) - TSFM: {mae_tsfm:.4f}")

# 5. Οπτικοποίηση Αποτελεσμάτων (Επίπεδο Β)
plt.figure(figsize=(12, 6))
plt.plot(results['portfolio_val_lstm'], label='Στρατηγική LSTM')
plt.plot(results['portfolio_val_tsfm'], label='Στρατηγική TSFM (Chronos)')
plt.title('Σύγκριση Απόδοσης Χαρτοφυλακίου (Backtest 2023-2024)')
plt.xlabel('Ημέρες')
plt.ylabel('Αξία Χαρτοφυλακίου (€)')
plt.legend()
plt.grid(True)
plt.show()
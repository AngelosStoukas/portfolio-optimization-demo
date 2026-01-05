import torch
from data_loader import get_market_data, to_tensor
from base_model import PortfolioOptimizer, sharpe_loss
from lstm_model import LSTMForecaster
from tsfm_model import TSFMForecaster

# 1. Παράμετροι και Δεδομένα
tickers = ['AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT']
returns_df = get_market_data(tickers)
returns_tensor = to_tensor(returns_df)

n_assets = len(tickers)
window_size = 30 # Οι τελευταίες 30 ημέρες για πρόβλεψη

# 2. Αρχικοποίηση Μοντέλων
# Ο Optimizer που βασίζεται στο αρχικό σου Neural Portfolio
optimizer = PortfolioOptimizer(input_size=n_assets, output_size=n_assets)

# Οι δύο Estimators (LSTM και TSFM)
lstm_forecaster = LSTMForecaster(input_dim=n_assets, hidden_dim=64)
tsfm_forecaster = TSFMForecaster()

# 3. Demo Εκτέλεση (Inference)
print("--- Εκτέλεση Πρόβλεψης & Βελτιστοποίησης ---")

# Παίρνουμε το τελευταίο "παράθυρο" δεδομένων για να προβλέψουμε το μέλλον
last_window = returns_tensor[-window_size:].unsqueeze(0) # [1, 30, 5]

# Πρόβλεψη από LSTM
lstm_pred = lstm_forecaster(last_window) # Προβλεπόμενα returns επόμενης μέρας

# Πρόβλεψη από TSFM (Chronos)
# Το Chronos θέλει transpose για να βλέπει κάθε μετοχή ως ξεχωριστή σειρά
tsfm_pred = tsfm_forecaster.predict(returns_tensor[-window_size:].T).unsqueeze(0)

# 4. Παραγωγή Βαρών Χαρτοφυλακίου
weights_lstm = optimizer(lstm_pred)
weights_tsfm = optimizer(tsfm_pred)

# 5. Εμφάνιση Αποτελεσμάτων
print("\nΒάρη Χαρτοφυλακίου (LSTM):")
for t, w in zip(tickers, weights_lstm[0].detach().numpy()):
    print(f"{t}: {w:.4f}")

print("\nΒάρη Χαρτοφυλακίου (TSFM/Chronos):")
for t, w in zip(tickers, weights_tsfm[0].detach().numpy()):
    print(f"{t}: {w:.4f}")
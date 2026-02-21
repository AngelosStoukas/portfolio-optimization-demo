import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Κατέβασμα Δεδομένων (Πιο ασφαλής τρόπος)
tickers = ['AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT']
# Κατεβάζουμε μόνο το 'Close' και το ισιώνουμε (flatten) αμέσως
data = yf.download(tickers, start="2020-01-01", end="2023-12-31")['Close']

# Υπολογισμός αποδόσεων
returns = data.pct_change().dropna()

# Μετατροπή σε Tensors για το PyTorch
X = torch.tensor(returns.values, dtype=torch.float32)

# 2. Το Βασικό ML Μοντέλο (Neural Portfolio)
class PortfolioNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PortfolioNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=1) # Άθροισμα βαρών = 1
        )
        
    def forward(self, x):
        return self.fc(x)

# 3. Custom Loss Function: Negative Sharpe Ratio
def sharpe_loss(weights, returns_tensor):
    # weights shape: [1, N], returns_tensor shape: [T, N]
    portfolio_return = torch.matmul(returns_tensor, weights.T)
    mean_return = torch.mean(portfolio_return)
    std_return = torch.std(portfolio_return)
    
    # Προσθήκη μικρής τιμής στον παρονομαστή για αποφυγή διαίρεσης με το μηδέν
    sharpe = mean_return / (std_return + 1e-6)
    return -sharpe 

# 4. Εκπαίδευση
input_n = len(tickers)
model = PortfolioNet(input_n, input_n)
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Starting training...")
for epoch in range(101):
    # Χρησιμοποιούμε όλο το ιστορικό για να μάθουμε τα ιδανικά βάρη
    weights = model(X.mean(dim=0).unsqueeze(0)) 
    loss = sharpe_loss(weights, X)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Negative Sharpe: {loss.item():.4f}")

# 5. Τελικά Βάρη
final_weights = model(X.mean(dim=0).unsqueeze(0)).detach().numpy()[0]
portfolio_results = dict(zip(tickers, np.round(final_weights, 4)))

print("\n--- ΑΠΟΤΕΛΕΣΜΑΤΑ ---")
print(f"Tickers: {tickers}")
print(f"Βέλτιστα Βάρη: {portfolio_results}")
print(f"Τελικό Sharpe Ratio: {-loss.item():.4f}")

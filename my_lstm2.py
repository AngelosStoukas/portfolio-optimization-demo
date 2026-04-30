import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf 

# 1. Προετοιμασία Δεδομένων
np.random.seed(0)
torch.manual_seed(0)

# Επιλογή 5 μετοχών για το πείραμα
tickers = ['AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT']
data = yf.download(tickers, start="2020-01-01", end="2023-12-31")

# Έλεγχος αν υπάρχει το 'Adj Close', αλλιώς χρησιμοποίησε το 'Close'
if 'Adj Close' in data.columns:
    df = data['Adj Close']
else:
    df = data['Close']

# Μετατροπή σε Log-Returns 
returns = np.log(df / df.shift(1)).dropna().values

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length] 
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 30
X, y = create_sequences(returns, seq_length)

# Διαστάσεις: (Samples, Seq_Length, N_Assets)
trainX = torch.tensor(X, dtype=torch.float32)
trainY = torch.tensor(y, dtype=torch.float32)

# 2. Ορισμός του PortfolioNet (LSTM + Softmax)
class PortfolioNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(PortfolioNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1) # Διασφαλίζει sum(weights) = 1

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Παίρνουμε την έξοδο του τελευταίου χρονικού βήματος
        out = self.fc(out[:, -1, :]) 
        weights = self.softmax(out) # Παραγωγή βαρών w
        return weights

# 3. Ορισμός της Συνάρτησης Sharpe Loss 
def sharpe_loss(weights, next_day_returns):
    # weights shape: (batch, n_assets)
    # next_day_returns shape: (batch, n_assets)
    portfolio_return = torch.sum(weights * next_day_returns, dim=1)
    
    mean_return = torch.mean(portfolio_return)
    std_return = torch.std(portfolio_return) + 1e-6 # ε=10^-6 για ευστάθεια
    
    # Επιστρέφουμε το αρνητικό Sharpe Ratio για ελαχιστοποίηση
    return -(mean_return / std_return)

# 4. Αρχικοποίηση και Εκπαίδευση
n_assets = len(tickers)
model = PortfolioNet(input_dim=n_assets, hidden_dim=64, layer_dim=2, output_dim=n_assets)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Το μοντέλο παράγει βάρη w*
    weights = model(trainX) 

    # Υπολογισμός Loss βάσει Sharpe (Επίπεδο 3)
    loss = sharpe_loss(weights, trainY) 
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Negative Sharpe Loss: {loss.item():.4f}')

# 5. Οπτικοποίηση Κατανομής Βαρών
model.eval()
final_weights = model(trainX[-1:].detach()) # Βάρη για την τελευταία ημέρα
final_weights = final_weights.detach().squeeze().numpy()

plt.figure(figsize=(10, 5))
plt.bar(tickers, final_weights, color='skyblue')
plt.title('Τελική Κατανομή Βαρών Χαρτοφυλακίου (LSTM Optimizer)')
plt.ylabel('Βάρος (w)')
plt.show()
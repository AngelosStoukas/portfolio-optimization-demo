import yfinance as yf
import torch
import pandas as pd

def get_market_data(tickers, start="2020-01-01", end="2023-12-31"):
    """Λήψη δεδομένων και υπολογισμός αποδόσεων."""
    data = yf.download(tickers, start=start, end=end)['Close']
    returns = data.pct_change().dropna()
    return returns

def to_tensor(df):
    """Μετατροπή DataFrame σε PyTorch Tensor."""
    return torch.tensor(df.values, dtype=torch.float32)
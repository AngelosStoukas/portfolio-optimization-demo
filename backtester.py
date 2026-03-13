import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_loader import get_market_data, to_tensor
from lstm_model import LSTMForecaster
from tsfm_model import TSFMForecaster
from base_model import PortfolioOptimizer, sharpe_loss, train_optimizer


# 1. ΠΑΡΑΜΕΤΡΟΙ
TICKERS     = ['AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT']
N_ASSETS    = len(TICKERS)
WINDOW_SIZE = 30
LSTM_EPOCHS = 15

# 2. LAYER 1 — ΔΕΔΟΜΕΝΑ

print(" Λήψη δεδομένων...")
all_data   = get_market_data(TICKERS, start="2020-01-01", end="2024-12-31")
all_tensor = to_tensor(all_data)

train_mask   = all_data.index < "2023-01-01"
train_data   = all_data[train_mask]
test_data    = all_data[~train_mask]
train_tensor = to_tensor(train_data)
test_tensor  = to_tensor(test_data)

print(f" Train: {train_data.index[0].date()} → {train_data.index[-1].date()}")
print(f" Test : {test_data.index[0].date()} → {test_data.index[-1].date()}")

# 3. LAYER 2 — ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΤΕΛΩΝ


# LSTM 
print("\n Εκπαίδευση LSTM...")
lstm      = LSTMForecaster(N_ASSETS, 64)
criterion = torch.nn.MSELoss()
lstm_opt  = torch.optim.Adam(lstm.parameters(), lr=0.001)

lstm.train()
for epoch in range(LSTM_EPOCHS):
    epoch_loss = 0
    for i in range(WINDOW_SIZE, len(train_tensor) - 1):
        x = train_tensor[i - WINDOW_SIZE:i].unsqueeze(0)
        y = train_tensor[i + 1]
        lstm_opt.zero_grad()
        pred = lstm(x)
        loss = criterion(pred, y.unsqueeze(0))
        loss.backward()
        lstm_opt.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/{LSTM_EPOCHS} | Loss: {epoch_loss/len(train_tensor):.6f}")

# TSFM (Chronos zero-shot) 
print("\n Φόρτωση Chronos...")
tsfm = TSFMForecaster()

#  Portfolio Optimizer 
print("\n Εκπαίδευση PortfolioOptimizer...")
optimizer_net = PortfolioOptimizer(N_ASSETS, N_ASSETS)
train_optimizer(optimizer_net, train_tensor, epochs=300)


# 4. BACKTESTING

print("\n📊 Backtesting...")

lstm_values  = [1.0]
tsfm_values  = [1.0]
equal_values = [1.0]
equal_w      = np.ones(N_ASSETS) / N_ASSETS

lstm_daily_rets  = []
tsfm_daily_rets  = []
equal_daily_rets = []

lstm.eval()
with torch.no_grad():
    for t in range(WINDOW_SIZE, len(test_tensor) - 1):
        context = test_tensor[t - WINDOW_SIZE:t]
        actual  = test_tensor[t + 1]

        p_lstm = lstm(context.unsqueeze(0))
        p_tsfm = tsfm.predict(context.T).unsqueeze(0)

        w_lstm = optimizer_net(p_lstm)
        w_tsfm = optimizer_net(p_tsfm)

        r_lstm  = torch.sum(w_lstm  * actual).item()
        r_tsfm  = torch.sum(w_tsfm  * actual).item()
        r_equal = float(np.sum(equal_w * actual.numpy()))

        lstm_values.append(lstm_values[-1]   * (1 + r_lstm))
        tsfm_values.append(tsfm_values[-1]   * (1 + r_tsfm))
        equal_values.append(equal_values[-1] * (1 + r_equal))

        lstm_daily_rets.append(r_lstm)
        tsfm_daily_rets.append(r_tsfm)
        equal_daily_rets.append(r_equal)


# 5. METRICS

def compute_metrics(daily_rets, label):
    r          = np.array(daily_rets)
    ann_return = np.mean(r) * 252
    ann_vol    = np.std(r)  * np.sqrt(252)
    sharpe     = ann_return / (ann_vol + 1e-6)
    equity     = np.cumprod(1 + r)
    peak       = np.maximum.accumulate(equity)
    max_dd     = ((equity - peak) / peak).min()

    print(f"\n {label}:")
    print(f"   Ann. Return : {ann_return:.2%}")
    print(f"   Volatility  : {ann_vol:.2%}")
    print(f"   Sharpe Ratio: {sharpe:.4f}")
    print(f"   Max Drawdown: {max_dd:.2%}")
    return {'label': label, 'ann_return': ann_return, 'ann_vol': ann_vol,
            'sharpe': sharpe, 'max_drawdown': max_dd, 'equity': equity}

print("\n" + "="*55)
r_lstm  = compute_metrics(lstm_daily_rets,  'LSTM')
r_tsfm  = compute_metrics(tsfm_daily_rets,  'TSFM (Chronos)')
r_equal = compute_metrics(equal_daily_rets, 'Equal Weight (Naive)')

# Πίνακας
summary = pd.DataFrame([r_lstm, r_tsfm, r_equal])
summary['ann_return']   = summary['ann_return'].map('{:.2%}'.format)
summary['ann_vol']      = summary['ann_vol'].map('{:.2%}'.format)
summary['sharpe']       = summary['sharpe'].map('{:.4f}'.format)
summary['max_drawdown'] = summary['max_drawdown'].map('{:.2%}'.format)
print("\n", summary[['label','ann_return','ann_vol','sharpe','max_drawdown']].to_string(index=False))

# 6. ΓΡΑΦΙΚΑ
colors = {'LSTM': '#2563EB', 'TSFM (Chronos)': '#DC2626', 'Equal Weight (Naive)': '#9CA3AF'}

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig)
fig.suptitle('Σύγκριση LSTM vs Chronos — Out-of-Sample 2023-2024',
             fontsize=14, fontweight='bold')

ax1 = fig.add_subplot(gs[0, :])
for res in [r_lstm, r_tsfm, r_equal]:
    ax1.plot(res['equity'], label=f"{res['label']} (Sharpe: {res['sharpe']:.2f})",
             color=colors[res['label']], linewidth=2)
ax1.axhline(1.0, color='black', linestyle='--', alpha=0.3)
ax1.set_title('Equity Curve')
ax1.set_ylabel('Portfolio Value (αρχικό = 1.0)')
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[1, 0])
sharpes = [m['sharpe'] for m in [r_lstm, r_tsfm, r_equal]]
labels  = [m['label'].split(' (')[0] for m in [r_lstm, r_tsfm, r_equal]]
bars = ax2.bar(labels, sharpes, color=[colors[m['label']] for m in [r_lstm, r_tsfm, r_equal]], alpha=0.85)
for bar, val in zip(bars, sharpes):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
ax2.set_title('Sharpe Ratio')
ax2.grid(alpha=0.3, axis='y')

ax3 = fig.add_subplot(gs[1, 1])
drawdowns = [abs(m['max_drawdown']) for m in [r_lstm, r_tsfm, r_equal]]
bars2 = ax3.bar(labels, drawdowns, color=[colors[m['label']] for m in [r_lstm, r_tsfm, r_equal]], alpha=0.85)
for bar, val in zip(bars2, drawdowns):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{val:.2%}', ha='center', fontsize=10, fontweight='bold')
ax3.set_title('Max Drawdown (χαμηλότερο = καλύτερο)')
ax3.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('comparison_results.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Αποθηκεύτηκε ως comparison_results.png")
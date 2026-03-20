# traditional_models.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize

from data_loader import get_market_data, to_tensor

# 1. ΠΑΡΑΜΕΤΡΟΙ (ίδιοι με backtester.py)

TICKERS     = ['AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT']
N_ASSETS    = len(TICKERS)
TRAIN_START = '2020-01-01'
TRAIN_END   = '2022-12-31'
TEST_START  = '2023-01-01'
TEST_END    = '2024-12-31'

# 2. LAYER 1 — ΔΕΔΟΜΕΝΑ (ίδιο με όλα τα υπόλοιπα)

print(' Λήψη δεδομένων...')
train_data = get_market_data(TICKERS, start=TRAIN_START, end=TRAIN_END)
test_data  = get_market_data(TICKERS, start=TEST_START,  end=TEST_END)

print(f' Train: {train_data.index[0].date()} → {train_data.index[-1].date()}')
print(f' Test : {test_data.index[0].date()} → {test_data.index[-1].date()}')

# 3. LAYER 2 — ΟΙ 3 ΚΛΑΣΙΚΕΣ ΜΕΘΟΔΟΙ

# Μέθοδος Α: Markowitz (Mean-Variance Optimization)
def markowitz_max_sharpe(returns_df, rf=0.0):
    """
    Κλασικό Markowitz
    Βρίσκει τα βάρη w που μεγιστοποιούν το Sharpe Ratio.
    Χρησιμοποιεί μόνο: μέσο απόδοσης (μ) και πίνακα συνδιακύμανσης (Σ).
    """
    mu  = returns_df.mean().values
    cov = returns_df.cov().values
    n   = len(mu)

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf) / (vol + 1e-8)

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds      = [(0.0, 1.0)] * n           # Long-only (χωρίς short selling)
    w0          = np.ones(n) / n             # Αρχή από equal weight

    result = minimize(neg_sharpe, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'ftol': 1e-12, 'maxiter': 1000})
    return result.x


# Μέθοδος Β: Risk Parity
def risk_parity(returns_df):
    """
    Risk Parity: κάθε μετοχή συνεισφέρει ίσο ρίσκο στο χαρτοφυλάκιο.
    Δεν χρειάζεται πρόβλεψη αποδόσεων — βασίζεται μόνο στη μεταβλητότητα.
    """
    cov = returns_df.cov().values
    n   = len(returns_df.columns)

    def risk_parity_obj(w):
        w       = np.array(w)
        sigma   = np.sqrt(w @ cov @ w)
        # Marginal Risk Contribution κάθε μετοχής
        mrc     = (cov @ w) / (sigma + 1e-8)
        # Risk Contribution = w * MRC
        rc      = w * mrc
        # Θέλουμε όλα τα RC να είναι ίσα → ελαχιστοποιούμε διαφορές
        target  = sigma / n
        return np.sum((rc - target) ** 2)

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds      = [(0.01, 1.0)] * n
    w0          = np.ones(n) / n

    result = minimize(risk_parity_obj, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'ftol': 1e-12, 'maxiter': 1000})
    return result.x


# Μέθοδος Γ: Equal Weight — 1/N κατανομή (Naive benchmark) 
def equal_weight(n_assets):
    """
    1/N κατανομή — το πιο απλό δυνατό benchmark.
    Αν κανένα μοντέλο δεν το νικά, κάτι πάει λάθος.
    """
    return np.ones(n_assets) / n_assets


# 4. LAYER 3 — ΥΠΟΛΟΓΙΣΜΟΣ ΒΑΡΩΝ ΑΠΟ TRAIN DATA

print('\n  Υπολογισμός βαρών...')

w_markowitz = markowitz_max_sharpe(train_data)
w_riskparity = risk_parity(train_data)
w_equal     = equal_weight(N_ASSETS)

print('\n Βάρη ανά μέθοδο:')
print(f"{'Ticker':<8} {'Markowitz':>12} {'Risk Parity':>12} {'Equal':>8}")
print('-' * 44)
for t, wm, wr, we in zip(TICKERS, w_markowitz, w_riskparity, w_equal):
    print(f'{t:<8} {wm:>11.2%} {wr:>11.2%} {we:>8.2%}')


# 5. LAYER 4 — BACKTESTING & METRICS (ίδια με backtester.py)

def compute_metrics(weights, test_df, label):
    """
    Εφαρμόζει σταθερά βάρη στα test data και υπολογίζει:
    Ann. Return, Volatility, Sharpe Ratio, Max Drawdown.
    """
    daily_rets = test_df.values @ weights        # ημερήσιες αποδόσεις χαρτοφυλακίου
    ann_return = np.mean(daily_rets) * 252
    ann_vol    = np.std(daily_rets)  * np.sqrt(252)
    sharpe     = ann_return / (ann_vol + 1e-8)

    equity     = np.cumprod(1 + daily_rets)
    peak       = np.maximum.accumulate(equity)
    max_dd     = ((equity - peak) / peak).min()

    print(f'\n {label}:')
    print(f'   Ann. Return : {ann_return:.2%}')
    print(f'   Volatility  : {ann_vol:.2%}')
    print(f'   Sharpe Ratio: {sharpe:.4f}')
    print(f'   Max Drawdown: {max_dd:.2%}')

    return {
        'label':       label,
        'weights':     weights,
        'ann_return':  ann_return,
        'ann_vol':     ann_vol,
        'sharpe':      sharpe,
        'max_drawdown':max_dd,
        'equity':      equity,
        'daily_rets':  daily_rets
    }

print('\n' + '='*55)
print('ΑΠΟΤΕΛΕΣΜΑΤΑ OUT-OF-SAMPLE (2023-2024)')
print('='*55)

r_markowitz  = compute_metrics(w_markowitz,  test_data, 'Markowitz (MVO)')
r_riskparity = compute_metrics(w_riskparity, test_data, 'Risk Parity')
r_equal      = compute_metrics(w_equal,      test_data, 'Equal Weight (1/N)')

# Πίνακας σύγκρισης (ίδια μορφή με backtester.py)
all_results = [r_markowitz, r_riskparity, r_equal]
summary = pd.DataFrame([{
    'Model':        r['label'],
    'Ann. Return':  f"{r['ann_return']:.2%}",
    'Volatility':   f"{r['ann_vol']:.2%}",
    'Sharpe Ratio': f"{r['sharpe']:.4f}",
    'Max Drawdown': f"{r['max_drawdown']:.2%}"
} for r in all_results])

print('\n' + '='*65)
print('ΠΙΝΑΚΑΣ ΣΥΓΚΡΙΣΗΣ — ΠΑΡΑΔΟΣΙΑΚΕΣ ΜΕΘΟΔΟΙ')
print('='*65)
print(summary.to_string(index=False))
print('='*65)



# 6. ΓΡΑΦΙΚΑ

colors = {
    'Markowitz (MVO)':    '#16A34A',
    'Risk Parity':        '#9333EA',
    'Equal Weight (1/N)': '#9CA3AF'
}

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig)
fig.suptitle('Παραδοσιακές Μέθοδοι Portfolio Optimization\nOut-of-Sample 2023-2024',
             fontsize=14, fontweight='bold')

# Equity Curves 
ax1 = fig.add_subplot(gs[0, :])
for r in all_results:
    ax1.plot(test_data.index[:len(r['equity'])], r['equity'],
             label=f"{r['label']} (Sharpe: {r['sharpe']:.2f})",
             color=colors[r['label']], linewidth=2)
ax1.axhline(1.0, color='black', linestyle='--', alpha=0.3, linewidth=1)
ax1.set_title('Equity Curve — Out-of-Sample')
ax1.set_ylabel('Portfolio Value (αρχικό = 1.0)')
ax1.legend()
ax1.grid(alpha=0.3)

# Βάρη ανά μέθοδο 
ax2 = fig.add_subplot(gs[1, 0])
x   = np.arange(N_ASSETS)
w   = 0.25
ax2.bar(x - w, w_markowitz,  width=w, label='Markowitz',   color='#16A34A', alpha=0.85)
ax2.bar(x,     w_riskparity, width=w, label='Risk Parity', color='#9333EA', alpha=0.85)
ax2.bar(x + w, w_equal,      width=w, label='Equal',       color='#9CA3AF', alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(TICKERS)
ax2.set_title('Κατανομή Βαρών ανά Μέθοδο')
ax2.set_ylabel('Βάρος')
ax2.legend()
ax2.grid(alpha=0.3, axis='y')

# Sharpe Ratio σύγκριση
ax3 = fig.add_subplot(gs[1, 1])
sharpes = [r['sharpe'] for r in all_results]
labels  = [r['label'] for r in all_results]
bars    = ax3.bar(labels, sharpes,
                  color=[colors[r['label']] for r in all_results], alpha=0.85)
for bar, val in zip(bars, sharpes):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
ax3.set_title('Sharpe Ratio Σύγκριση')
ax3.set_ylabel('Sharpe Ratio')
ax3.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('traditional_results.png', dpi=150, bbox_inches='tight')
plt.show()
print('\n Αποθηκεύτηκε ως traditional_results.png')

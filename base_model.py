import torch
import torch.nn as nn

class PortfolioOptimizer(nn.Module):
    def __init__(self, input_size, output_size):
        super(PortfolioOptimizer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=1) # Διασφαλίζει άθροισμα βαρών = 1
        )
        
    def forward(self, x):
        return self.fc(x)

def sharpe_loss(weights, returns_tensor):
    """Υπολογισμός Negative Sharpe Ratio."""
    portfolio_return = torch.matmul(returns_tensor, weights.T)
    mean_return = torch.mean(portfolio_return)
    std_return = torch.std(portfolio_return)
    sharpe = mean_return / (std_return + 1e-6)
    return -sharpe
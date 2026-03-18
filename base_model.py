import torch
import torch.nn as nn
import torch.optim as optim


class PortfolioOptimizer(nn.Module):
    def __init__(self, input_size, output_size):
        super(PortfolioOptimizer, self).__init__()
        self.fc1     = nn.Linear(input_size, 64)
        self.fc2     = nn.Linear(64, 32)
        self.fc3     = nn.Linear(32, output_size)
        self.relu    = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


def sharpe_loss(weights, returns_tensor, rf=0.0):
    portfolio_return = torch.matmul(returns_tensor, weights.T).squeeze()
    mean_r = torch.mean(portfolio_return)
    std_r  = torch.std(portfolio_return)
    sharpe = (mean_r - rf) / (std_r + 1e-6)
    return -sharpe


def train_optimizer(model, returns_tensor, epochs=500, lr=0.01):
    optimizer  = optim.Adam(model.parameters(), lr=lr)
    mean_input = returns_tensor.mean(dim=0).unsqueeze(0)

    print("Εκπαίδευση PortfolioOptimizer...")
    for epoch in range(epochs + 1):
        weights = model(mean_input)
        loss    = sharpe_loss(weights, returns_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d} | Sharpe: {-loss.item():.4f}")

    return model(mean_input).detach().numpy()[0]

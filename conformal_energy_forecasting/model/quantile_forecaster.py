import torch
import torch.nn as nn


class QuantileForecaster(nn.Module):
    """
    Multi-output quantile regression MLP for energy demand forecasting.
    Produces three outputs per input: lower quantile, median, upper quantile.

    Input:  (batch, n_features)
    Output: (batch, 3) — columns are [q_lower, q_median, q_upper]
    """

    def __init__(self, n_features: int, hidden1: int = 128, hidden2: int = 64,
                 dropout: float = 0.2):
        super(QuantileForecaster, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head_lower  = nn.Linear(hidden2, 1)
        self.head_median = nn.Linear(hidden2, 1)
        self.head_upper  = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        q_lower  = self.head_lower(h)
        q_median = self.head_median(h)
        q_upper  = self.head_upper(h)
        return torch.cat([q_lower, q_median, q_upper], dim=1)

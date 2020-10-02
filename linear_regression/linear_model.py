import torch.nn as nn


class ManualRegression(nn.Module):

    def __init__(self):
        super().__init__()
        # self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        # self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.linear = nn.Linear(1, 1)  # basic linear pytoch layer

    def forward(self, x):
        return self.linear(x)

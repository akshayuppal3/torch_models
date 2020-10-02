import torch.nn as nn
import torch.nn.functional as F
import torch


class SimpleMLP(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SimpleMLP, self).__init__()
        self.d_ff = d_model
        self.fc11 = nn.Linear(d_model, self.d_ff)
        self.fc2 = nn.Linear(self.d_ff, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.fc11(x)
        activated = F.relu(hidden)
        dropout = self.dropout(activated)
        output = self.fc2(dropout)
        return torch.relu(torch.sign(output))  # not ideal but did for specific dataset

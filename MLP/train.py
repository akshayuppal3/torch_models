import torch
from pathlib import Path
from tqdm import trange
# import copy

from torch import nn
import torch.optim as optim

from MLP.model import SimpleMLP
from MLP.preprocess import get_data_text
from common_utils.util import get_train_val_loader

data_path = Path("MLP/data/ret_label_x_y.pkl")

if __name__ == '__main__':
    # cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)

    x_tensor, y_tensor = get_data_text(data_path)

    train_loader = get_train_val_loader(x_tensor, y_tensor, shuffle=False)

    lr = 0.1
    n_epochs = 100
    model = SimpleMLP(x_tensor.size()[-1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(model.state_dict())

    loss_fn = nn.BCELoss()

    best_score = -1
    best_loss = float('inf')
    best_model, best_report = None, None

    for epoch in trange(n_epochs):
        train_loss = 0.0
        val_loss = 0.0
        for x_feat, y_labels in train_loader:
            x_feat = x_feat.to(device)
            y_labels = y_labels.to(device)

            model.train()

            # compute the loss
            pr_labels = torch.flatten(model(x_feat))
            loss = loss_fn(pr_labels, y_labels)

            # compute the graients
            loss.sum().backward()
            train_loss += loss.sum().item()

            # update the params
            optimizer.step()
            optimizer.zero_grad()

        print("train_loss", train_loss, "epoch",epoch)
    print(model.state_dict())

import torch
from pathlib import Path
from tqdm import trange
# import copy

from torch import nn
import torch.optim as optim

from model import SimpleMLP
from preprocess import get_data_text
from common_utils.util import get_train_val_loader

data_path = Path("MLP/data/ret_label_x_y.pkl")


if __name__ == '__main__':
    x_tensor, y_tensor = get_data_text(data_path)
    print(x_tensor, y_tensor)


def main():
    # cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)

    x_tensor, y_tensor = get_data_text(data_path)

    train_loader, val_loader = get_train_val_loader(x_tensor, y_tensor, shuffle=False)

    lr = 0.01
    n_epochs = 1000
    model = SimpleMLP(x_tensor.size()[-1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(model.state_dict())

    loss_fn = nn.CrossEntropyLoss()

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
            pr_labels = model(x_feat.float())
            loss = loss_fn(pr_labels, y_labels)

            # compute the graients
            loss.sum().backward()
            train_loss += loss.sum().item()

            # update the params
            optimizer.step()
            optimizer.zero_grad()

        # evaluate on the dev sample
        with torch.no_grad(): # make sure gradient is not cal in evaluation phase
            if val_loader is not None:
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)

                    model.eval()

                    yhat = model(x_val)

                    val_loss = loss_fn(y_val, yhat)
                    val_loss += val_loss.item()

                print("epoch = ", epoch, "val_loss", val_loss, "train_loss", train_loss)

            #         report = eval_model(model, x_val, y_val)
            #         if best_score < report['1']['f1_score']:
            #             best_score = report['1']['f1_score']
            #             best_report = report
            #             best_model = copy.deepcopy(model)
            #
            # else:
            #     if best_loss > train_loss:
            #         best_loss = train_loss
            #         best_model = copy.deepcopy(model)

    # print(best_model)
    print(model.state_dict())

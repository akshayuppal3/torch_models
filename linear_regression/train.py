import torch
import torch.nn as nn
import torch.optim as optim

from common_utils import get_train_val_loader
from linear_model import ManualRegression
from preprocessing import make_train_step, get_data

if __name__ == '__main__':

    # switch to cuda if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(42)

    lr = 0.01
    n_epochs = 1000
    model = ManualRegression().to(device)

    print(model.state_dict())

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=lr)  # could use Adam optimizer as well

    x_tensor, y_tensor = get_data()
    train_loader, val_loader = get_train_val_loader(x_tensor, y_tensor, shuffle=False)

    train_step = make_train_step(model, loss_fn, optimizer)

    losses = []
    val_losses = []
    for epoch in range(n_epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            loss = train_step(x_batch, y_batch)

            losses.append(loss)

        # evaluation on val data
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)  # loading only sample of dataset in memory
                y_val = y_val.to(device)

                model.eval()

                yhat = model(x_val)

                val_loss = loss_fn(y_val, yhat)
                val_losses.append(val_loss.item())

            print("epoch = ", epoch, "val_loss", sum(val_losses) / (epoch + 1), "train_loss", sum(losses) / (epoch + 1))

    # check model params
    print(model.state_dict())

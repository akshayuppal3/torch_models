import torch
import torch.nn as nn
import torch.optim as optim

from linear_model import ManualRegression
from preprocessing import get_x_y

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)

    lr = 0.05
    n_epochs = 1000
    model = ManualRegression().to(device)

    print(model.state_dict())

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=lr)

    x_train, y_train = get_x_y()

    for epoch in range(n_epochs):
        model.train()

        # prediction
        yhat = model(x_train)

        # loss
        # loss = -2 * (error ** 2).mean(
        loss = loss_fn(y_train, yhat)

        # SGD
        # a_grad = -2 * error.mean()
        # b_grad = -2 * (x_train * error).mean()
        loss.backward()

        # updaye
        # a = a - lr * a_grad
        # b = b = lr * b_grad
        optimizer.step()

        # pytocrh requires to reset the gradients
        optimizer.zero_grad()

    print(model.state_dict())

    # evaluate the performance

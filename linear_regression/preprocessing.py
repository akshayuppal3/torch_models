import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def get_data(x_dim=100):
    x = np.random.rand(x_dim, 1)
    assert x.shape == (x_dim, 1)

    # linear equation
    y = 1 + 2 * x + 0.1 * np.random.rand(100, 1)

    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    return x_tensor, y_tensor


def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        # train model
        model.train()

        # make prediction
        yhat = model(x)

        # compute the loss
        loss = loss_fn(y, yhat)

        # compute the gradients
        loss.sum().backward()

        # Update the parametes and zero gradients
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    # return function that will be called inside train loop
    return train_step

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split, TensorDataset


class CustomeDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


def get_train_val_loader(train_batch_size=16, val_batch_size=20, shuffle=True):
    x = np.random.rand(100, 1)
    assert x.shape == (100, 1)

    # linear equation
    y = 1 + 2 * x + 0.1 * np.random.rand(100, 1)

    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    dataset = CustomeDataset(x_tensor, y_tensor)  # could use TensorDataset

    train_dataset, val_dataset = random_split(dataset, [80, 20])

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=shuffle)
    val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=shuffle)

    return train_loader, val_loader


def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        # train model
        model.train()

        # make prediction
        yhat = model(x)

        # compute the loss
        loss = loss_fn(y, yhat)

        # compute the gradients
        loss.backward()

        # Update the parametes and zero gradients
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    # return function taht will be called inside train loop
    return train_step

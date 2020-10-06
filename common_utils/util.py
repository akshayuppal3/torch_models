import numpy as np
import torch

from torch.utils.data import random_split, SequentialSampler, DataLoader, Dataset


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


def get_train_val_loader(
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        split_seq=[80, 20],
        train_batch_size=16,
        val_batch_size=20,
        shuffle=True
):
    # useful if want to use DataLoader that will load data in mini batches
    dataset = CustomDataset(x_tensor, y_tensor)  # could also use TensorDataset

    train_dataset, val_dataset = random_split(dataset, split_seq)

    train_sampler = SequentialSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    # useful for getting minibatch of data (mini-batch gradient descent)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=val_batch_size, shuffle=shuffle)

    return train_loader, val_loader

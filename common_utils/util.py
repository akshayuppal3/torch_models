import torch

from torch.utils.data import random_split, SequentialSampler, DataLoader

from preprocessing import CustomDataset


def get_train_val_loader(
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        train_batch_size=16,
        val_batch_size=20,
        shuffle=True
):
    # useful if want to use DataLoader that will load data in mini batches
    dataset = CustomDataset(x_tensor, y_tensor)  # could also use TensorDataset

    train_dataset, val_dataset = random_split(dataset, [80, 20])

    train_sampler = SequentialSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    # useful for getting minibatch of data (mini-batch gradient descent)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=val_batch_size, shuffle=shuffle)

    return train_loader, val_loader

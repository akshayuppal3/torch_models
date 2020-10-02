import torch
import numpy as np


def get_x_y():
    x = np.random.rand(100, 1)
    assert x.shape == (100, 1)

    # linear equation
    y = 1 + 2 * x + 0.1 * np.random.rand(100, 1)

    # shuffle the indices
    idx = np.arange(100)
    np.random.shuffle(idx)

    train_idx = idx[:80]
    test_idx = idx[80:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    # switch to cuda if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)

    return x_train_tensor, y_train_tensor

import pickle
from pathlib import Path

import torch


def get_data_text(file_path: Path):
    with file_path.open('rb') as fopen:
        label_x_y= pickle.load(fopen)
    x, y = label_x_y[0][0], label_x_y[0][1]
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    return x_tensor, y_tensor

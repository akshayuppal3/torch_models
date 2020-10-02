import pickle
from pathlib import Path

import torch


def get_data_text(file_path: Path):
    label, x, y = pickle.load(file_path)
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    return x_tensor, y_tensor

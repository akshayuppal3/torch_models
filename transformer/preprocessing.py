import pickle
from pathlib import Path

import torch


def get_token_data(file_path: Path, filename: str):
    full_path = file_path.joinpath(filename)
    with full_path.open('rb') as fin:
        x_y_label_vocab = pickle.load(fin)
    data = x_y_label_vocab[0]
    X, y, label, vocab = data['X'], data['y'], data['label'], data['vocab']
    return X, y, data['label'], data['vocab']

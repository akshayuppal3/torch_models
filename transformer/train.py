import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, random_split
from tqdm import trange

from transformer.model import TransformerNetwork
from transformer.preprocessing import get_classification_report, __get_token_data, get_data_loader

data_dir = Path("transformer/data/")


def train(
        max_vocab_size: int,
        train_data: TensorDataset,
        batch_size: int = 32,
        lr: float = 0.001,
        epochs: int = 10,
        head_num: int = 4,
        dropout: float = 0.001,
        embed_dim: int = 512,
        dev: TensorDataset = None
):
    # switch to cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader = get_data_loader(
        train_data,
        batch_size=batch_size
    )
    model = TransformerNetwork(vocab_size=max_vocab_size, head_num=head_num, dropout=dropout, embed_dim=embed_dim)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fct = nn.CrossEntropyLoss()

    best_loss = 1.0
    best_score, best_report, best_model = 0.0, None, None
    for _ in trange(epochs, desc='Epoch'):
        train_loss = 0.0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_loader):
            if torch.cuda.is_available():
                batch = tuple(item.to(device) for item in batch)
            bt_features, bt_label = batch
            print("feat", bt_features.size())
            print("label", bt_label.size())
            pr_labels = model(bt_features).view(-1, 2)
            loss = loss_fct(pr_labels, bt_label.view(-1))
            loss.sum().backward()
            train_loss += loss.sum().item()
            nb_tr_examples += bt_features.size(0)
            nb_tr_steps += 1
            optimizer.step()
            optimizer.zero_grad()

        print("epoch = ", nb_tr_steps, "train_loss", train_loss / nb_tr_steps)

        # evaluation
        if dev is not None:
            with torch.no_grad():
                report = get_classification_report(model, dev)
                print(report)
                if '1' in report and best_score < report['1']['f1-score']:
                    best_score = report['1']['f1-score']
                    best_model = copy.deepcopy(model)
        else:
            if train_loss / nb_tr_steps < best_loss:
                best_loss = train_loss / nb_tr_steps
                best_model = copy.deepcopy(model)

        print(best_score)

    return best_model if best_model else model


if __name__ == '__main__':
    X, y = __get_token_data(file_path=data_dir)
    x_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    dataset = TensorDataset(x_tensor, y_tensor)
    train_data_len = int(0.8 * len(dataset))
    print("train data len =", train_data_len)
    split_seq = [train_data_len, len(dataset) - train_data_len]

    train_data, val_data = random_split(dataset, split_seq)
    train(
        max_vocab_size=3000 + 2,
        train_data=train_data,
        head_num=4,
        epochs=10,
        dropout=0.001,
        embed_dim=512,
        dev=val_data
    )

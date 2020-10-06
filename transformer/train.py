import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange

from transformer.model import MyNetwork
from util import get_train_val_loader


def train(vocab_size, tr_features, tr_labels, batch_size=32, lr=0.001, epochs=10, dev=None):
    # switch to cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_tensor = torch.from_numpy(tr_features).float()
    y_tensor = torch.from_numpy(tr_labels).float()

    train_loader, val_loader = get_train_val_loader(x_tensor, y_tensor, train_batch_size=batch_size)

    model = MyNetwork(vocab_size=vocab_size)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fct = nn.CrossEntropyLoss()

    best_loss = 1.0
    best_score, best_report, best_model = 0.0, None, None

    for _ in trange(epochs, desc='Epoch'):
        train_loss = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_loader):
            if torch.cuda.is_available():
                batch = tuple(item.cuda() for item in batch)
            bt_features, bt_label = batch
            pr_labels = model(bt_features).view(-1,2)
            loss = loss_fct(pr_labels, bt_label.view(-1))
            loss.sum.backward()
            train_loss += loss.sum.item()
            nb_tr_examples += bt_features.size(0)
            nb_tr_steps += 1
            optimizer.step()
            optimizer.zero_grad()

            print("epoch = ", nb_tr_steps, "train_loss", train_loss / nb_tr_steps )

        # if dev is not None:
        #     with torch.no_grad():
        #         pass

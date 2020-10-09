import pickle
import torch
from pathlib import Path

from sklearn.metrics import classification_report

from torch.utils.data import TensorDataset, SequentialSampler, DataLoader


def __get_pkl_data(file_path, filename: str):
    full_path = file_path.joinpath(filename)
    obj = pickle.load(full_path.open('rb'))
    return obj


def __get_token_data(file_path: Path) -> (np.array, np.array):
    X = __get_pkl_data(file_path, 'X.pkl')
    y = __get_pkl_data(file_path, 'y.pkl')
    return X, y


def get_data_loader(
        dataset: TensorDataset,
        batch_size=32,
):
    train_sampler = SequentialSampler(dataset)

    # useful for getting minibatch of data (mini-batch gradient descent)
    data_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return data_loader


def get_classification_report(model, dev_dataset: TensorDataset, batch_size=32, output_form=True):
    data_loader = get_data_loader(dev_dataset, batch_size=batch_size)

    true_labels, predicted_labels = list(), list()

    model.eval()

    for step, batch in enumerate(data_loader):
        if torch.cuda.is_available():
            batch = tuple(item.cuda() for item in batch)
        bt_features, bt_labels = batch
        pr_labels = model(bt_features)
        # we have
        x = [tensor_label.item() for batch in bt_labels for tensor_label in batch]
        y = [torch.argmax(tensor_label).item() for batch in pr_labels for tensor_label in batch]
        true_labels.extend(x)
        predicted_labels.extend(y)

    return classification_report(true_labels, predicted_labels, output_dict=output_form, digits=4)

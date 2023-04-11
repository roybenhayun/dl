import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from tqdm import tqdm

#
# Dataset class
#


class DiabetesDataset(Dataset):
    """
    encapsulates the sklear diabetes dataset
    see
        https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html for example implementation
    """

    def __init__(self, file):
        """
        load the CSV to Panda frame
        """
        self._csv_df = pd.read_csv(file, sep='\t')
        min_label = self._csv_df["Y"].min()
        self._csv_df['Label'] = self._csv_df['Y'] - min_label

        def df_to_tensor(df):
            return torch.from_numpy(df.values).float()

        self._transform = df_to_tensor

    def __len__(self) -> int:
        return len(self._csv_df.index)

    def __getitem__(self, idx) -> tuple:
        return self._transform(self._csv_df.iloc[idx, ~self._csv_df.columns.isin(['Y', 'Label'])]), \
            self._csv_df.iloc[idx, self._csv_df.columns == 'Label'].item()

    def get_min_label(self) -> float:
        return self._csv_df["Y"].min()

    def get_labels_range(self) -> tuple:
        return self._csv_df["Y"].min(), self._csv_df["Y"].max()

#
# Operations
#

def iterate_batch(input_tensor, labels, optimizer, model, CE_loss):
    optimizer.zero_grad()
    y_model = model(input_tensor)

    loss = CE_loss(y_model, labels.long())  # must accept long, and Target < model output_size
    loss.backward()
    optimizer.step()

    predicted_labels = y_model.argmax(dim=1)
    acc = (predicted_labels == labels).sum() / len(labels)
    return loss.detach(), acc.detach()


def validate_diabetes_ds(ds):
    assert len(ds) == 442, f"expected 442, got {len(ds)}"
    min, max = ds.get_labels_range()
    assert min == 25 and max == 346, r'expected 25 and 346 as in https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes'
    features, target = ds[0]
    assert len(features) == 10
    assert features[0] == 59 and target == 151 - ds.get_min_label(), f"expected 59 and {151 - ds.get_min_label()}, but received {features[0]}, {target}"
    features, target = ds[1]
    assert len(features) == 10
    assert features[0] == 48 and target == 75 - ds.get_min_label(), f"expected 48 and {75 - ds.get_min_label()}, but received {features[0]}, {target}"
    features, target = ds[441]
    assert len(features) == 10
    assert features[0] == 36 and target == 57 - ds.get_min_label(), f"expected 36 and {57 - ds.get_min_label()}, but received {features[0]}, {target}"
    features, target = ds[-1]
    assert len(features) == 10
    assert features[0] == 36 and target == 57 - ds.get_min_label(), f"expected 36 and {57 - ds.get_min_label()} , but received {features[0]}, {target}"



def render_accuracy_plot(batches, loss, acc, title):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(range(batches), loss)
    plt.title("CE loss")
    plt.xlabel("Batch Number")
    plt.subplot(1,2,2)
    plt.plot(range(batches), acc)
    avg_acc = round(float(np.average(acc)), 3)
    plt.title(f"Accuracy (avg: {avg_acc})")
    plt.xlabel("Batch Number")
    plt.suptitle(title)
    plt.show()


def train_and_test_subsets(test_set_size, train_set_size, dataset, num_epochs):
    subsets = random_split(dataset, [train_set_size, test_set_size])
    print(f"train set: {len(subsets[0])}, test set: {len(subsets[1])}")

    min, max = dataset.get_labels_range()
    num_labels = max - min + 1
    model = nn.Sequential(nn.Linear(10, num_labels),
                          nn.LogSoftmax(dim=1))
    CE_loss = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_dataloader = DataLoader(subsets[0], batch_size=10, shuffle=True)
    train_set_loss = torch.zeros(len(train_dataloader) * num_epochs)
    train_set_acc = torch.zeros(len(train_dataloader) * num_epochs)
    print("run train set..")

    # train model
    model.train()

    idx = 0
    for epoch in tqdm(range(num_epochs), unit="epoch"):
        for batch_idx, (features, labels) in enumerate(train_dataloader):
            train_set_loss[idx], train_set_acc[idx] = iterate_batch(features, labels, optimizer, model, CE_loss)
            idx += 1
    print(f"avg loss: {round(float(np.average(train_set_loss)), 3)}, avg acc: {round(float(np.average(train_set_acc)), 3)}")

    # test model

    test_dataloader = DataLoader(subsets[1], batch_size=10, shuffle=True)
    test_set_loss = torch.zeros(len(test_dataloader))
    test_set_acc = torch.zeros(len(test_dataloader))
    print("run test set..")
    model.eval()
    with torch.no_grad():
        for batch_idx, (features, labels) in tqdm(enumerate(test_dataloader), unit="test batch iteration"):
            test_set_loss[batch_idx], test_set_acc[batch_idx] = get_test_batch_accuracy(features, labels, model, CE_loss)
    print(f"avg loss: {round(float(np.average(test_set_loss)), 3)}, avg acc: {round(float(np.average(test_set_acc)), 3)}")

    return len(train_dataloader) * num_epochs, train_set_loss, train_set_acc, \
        len(test_dataloader), test_set_loss, test_set_acc


def get_test_batch_accuracy(input_tensor, labels, model, CE_loss):
    """
    similar operation as iterate_batch() without Gradient calculation and Optimizer operations, for benchmarking only
    """
    y_model = model(input_tensor)

    loss = CE_loss(y_model, labels.long())

    predicted_labels = y_model.argmax(dim=1)
    acc = (predicted_labels == labels).sum() / len(labels)
    return loss.detach(), acc.detach()


def render_train_test_accuracy_plot(num_train_batches, train_loss, train_acc,
                                    num_test_batches, test_loss, test_acc,
                                    title):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(range(num_train_batches), train_loss, label='train set')
    plt.title("CE loss")
    plt.xlabel("Batch Number")
    plt.legend()
    x2 = np.arange(num_train_batches, num_test_batches + num_train_batches)
    plt.plot(x2, test_loss, label='test set')
    min_test_ce_loss = round(float(test_loss.min()), 3)
    max_test_ce_loss = round(float(test_loss.max()), 3)
    avg_test_ce_loss = round(float(np.average(test_loss)), 3)
    plt.title(f"CE loss (min: {min_test_ce_loss}, max: {max_test_ce_loss}, avg: {avg_test_ce_loss})")
    plt.xlabel("Batch Number")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(num_train_batches), train_acc, label='train set')
    plt.title("Accuracy", color='orange')
    plt.xlabel("Batch Number")
    plt.legend()
    plt.plot(x2, test_acc, label='test set')
    min_test_acc = round(float(test_acc.min()), 3)
    max_test_acc = round(float(test_acc.max()), 3)
    avg_test_acc = round(float(np.average(test_acc)), 3)
    plt.title(f"Accuracy (min: {min_test_acc}, max: {max_test_acc}, avg: {avg_test_acc})")
    plt.xlabel("Batch Number")
    plt.legend()
    plt.suptitle(title)
    plt.show()


if __name__ == '__main__':
    csv_file = os.path.join(os.path.dirname(__file__), "diabetes.csv")
    if not os.path.exists(csv_file):
        raise EnvironmentError(f'{csv_file} not found')
    ds = DiabetesDataset(csv_file)
    validate_diabetes_ds(ds)

    print("train and test...")
    train_set_size = int(len(ds) * 0.8)
    test_set_size = len(ds) - train_set_size
    render_train_test_accuracy_plot(*train_and_test_subsets(test_set_size, train_set_size, ds, 100),
                                    "Diabetes, predict Y as Target")

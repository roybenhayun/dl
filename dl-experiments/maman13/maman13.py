import torch
import sklearn.datasets as skds
import numpy as np
from pandas.core.array_algos import transforms
from torch.utils.data import Dataset, DataLoader
import os
from matplotlib import pyplot as plt

# info on dataset:
#   https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
#   https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
#
# info on API
#   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes
#       Bunch.data: (442, 10)
#       Bunch.feature_names: list

bunch = skds.load_diabetes(return_X_y=False)


#   X: data (442, 10)
#   Y: regression target (442)
X, Y = skds.load_diabetes(return_X_y=True)


# 1C
# calc the ten deciles
# see https://www.geeksforgeeks.org/quantile-and-decile-rank-of-a-column-in-pandas-python/

# importing the modules
import pandas as pd

df = {'Name': ['Amit', 'Darren', 'Cody', 'Drew',
               'Ravi', 'Donald', 'Amy'],
      'Score': [50, 71, 87, 95, 63, 32, 80]}
df = pd.DataFrame(df, columns=['Name', 'Score'])

# adding Decile_rank column to the DataFrame
df['Quantile_rank'] = pd.qcut(df['Score'], 4, labels=["Q1", "Q2", "Q3", "Q4"])
sdf = df.sort_values(by=['Quantile_rank'])
#print(sdf)

df['Decile_rank'] = pd.qcut(df['Score'], 10, labels=False)
sdf = df.sort_values(by=['Quantile_rank', 'Decile_rank'])
#print(sdf)


# 1C
y_df = pd.DataFrame(Y, columns=['Score'])
decile_labels = [*range(1, 11)]
decile_labels.reverse()
y_df['Decile_rank'] = pd.qcut(y_df['Score'], 10, labels=decile_labels)
# y_sdf = y_df.sort_values(by=['Decile_rank'])
# print(y_sdf)


# 1D
# add Class: decile to X
x_df = pd.DataFrame(X, columns=bunch.feature_names)  # NOTE: this X created from sklearn load_diabetes - Y is NOT included
x_df['Class'] = y_df['Decile_rank']

# test two cases with min and max values in Y
min_idx_in_y = np.where(Y == Y.min())
max_idx_in_y = np.where(Y == Y.max())
assert x_df.iloc[min_idx_in_y].iloc[0]['Class'] == 10, f"{Y.min()} should be Class 10 in x_df"
assert x_df.iloc[max_idx_in_y].iloc[0]['Class'] == 1, f"{Y.max()} should be Class 1 in x_df"
print(f"max value in Y: {Y.max()}, in index: {max_idx_in_y}, has 'Class' value: {x_df.iloc[max_idx_in_y].iloc[0]['Class']} in x_df")
print(f"x_df[{max_idx_in_y[0][0]}] = \n{x_df.iloc[max_idx_in_y]}")
print(f"min value in Y: {Y.min()}, in index: {min_idx_in_y}, has 'Class' value: {x_df.iloc[min_idx_in_y].iloc[0]['Class']} in x_df")
print(f"x_df[{min_idx_in_y[0][0]}] = \n{x_df.iloc[min_idx_in_y]}")

# 1E


class DiabetesDataset(Dataset):
    """
    encapsulates the sklear diabetes dataset
    see
        https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html for example implementation
    """

    def __init__(self, file, with_y=True):
        """
        load the CSV to Panda frame
        """
        self._csv_df = pd.read_csv(file, sep='\t')
        self._with_Y = with_y
        # NOTE: Y is last column in CSV
        decile_labels = [*range(1, 11)]
        decile_labels.reverse()
        self._csv_df['Target'] = pd.qcut(pd.DataFrame(self._csv_df, columns=['Y'])['Y'], 10, labels=decile_labels)

        def df_to_tensor(df):
            return torch.from_numpy(df.values).float()

        self._transform = df_to_tensor

    def __len__(self) -> int:
        """
        return the size of the dataset
        """
        return len(self._csv_df.index)

    def __getitem__(self, idx) -> tuple:
        """
        fetching a data sample for a given key
        """
        # TODO: w\wo Y, return Class as Target
        if self._with_Y:
            return self._transform(self._csv_df.iloc[idx, self._csv_df.columns != 'Target']),\
                self._csv_df.iloc[idx, self._csv_df.columns == 'Target'].item()
        else:
            return self._transform(self._csv_df.iloc[idx, ~self._csv_df.columns.isin(['Y', 'Target'])]),\
                self._csv_df.iloc[idx, self._csv_df.columns == 'Target'].item()


csv_file = os.path.join(os.path.dirname(__file__), "diabetes.csv")  # NOTE: Y included in the CSV
if not os.path.exists(csv_file):
    raise EnvironmentError(f'{csv_file} not found')

diabetes_ds = DiabetesDataset(csv_file, with_y=False)
print(f"number of rows: {len(diabetes_ds)}")
assert len(diabetes_ds) == 442, f"expected 442, got {len(diabetes_ds)}"
features, target = diabetes_ds[0]
print(f"first row: features: {features}, size: {features.size()}, target: {target}")
features, target = diabetes_ds[1]
print(f"second row: features: {features}, size: {features.size()}, target: {target}")
features, target = diabetes_ds[2]
print(f"third row: features: {features}, size: {features.size()}, target: {target}")
features, target = diabetes_ds[441]
print(f"last row: features: {features}, size: {features.size()}, target: {target}")
features, target = diabetes_ds[-1]
print(f"last row: features: {features}, size: {features.size()}, target: {target}")

diabetes_ds = DiabetesDataset(csv_file, with_y=True)
print(f"number of rows: {len(diabetes_ds)}")
assert len(diabetes_ds) == 442, f"expected 442, got {len(diabetes_ds)}"
features, target = diabetes_ds[0]
print(f"first row: features: {features}, size: {features.size()}, target: {target}")
features, target = diabetes_ds[1]
print(f"second row: features: {features}, size: {features.size()}, target: {target}")
features, target = diabetes_ds[2]
print(f"third row: features: {features}, size: {features.size()}, target: {target}")
features, target = diabetes_ds[441]
print(f"last row: features: {features}, size: {features.size()}, target: {target}")
features, target = diabetes_ds[-1]
print(f"last row: features: {features}, size: {features.size()}, target: {target}")

# 1.6, 1.7

diabetes_ds = DiabetesDataset(csv_file, with_y=True)
ds_loader = DataLoader(diabetes_ds, batch_size=10, shuffle=False)

for features, labels in ds_loader:
    print(f"{features}, {labels}")

ds_iter = iter(ds_loader)
enumerator = enumerate(ds_iter)
for batch_idx, (features, labels) in enumerator:
    print(f"{batch_idx}, {features}, {labels}")

# iterable
ds_iter = iter(ds_loader)
print(f"batch_train_features: {ds_iter}")
batch_features, batch_labels = next(ds_iter)
print(batch_features, batch_labels)


# 1.8 - החוזה את Class על סמך שאר המשתנים

from torch import nn

include_Y = True
input_tensor_size = 11 if include_Y else 10  # the labels are the 11 deciles or 10 deciles w\wo Y respectively
output_size = 10  # number of Target labels
model = nn.Sequential(nn.Linear(input_tensor_size, output_size),  # input size, output size
                      nn.LogSoftmax(dim=1))  # activation on the output column
print(model)
CE_loss = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def iterate_batch(input_tensor, labels):
    optimizer.zero_grad()
    y_model = model(input_tensor)

    loss = CE_loss(y_model, labels.long() - 1)  # must accept long, and Target < model output_size
    loss.backward()
    optimizer.step()

    predicted_labels = y_model.argmax(dim=1)
    acc = (predicted_labels == labels).sum() / len(labels)
    return loss.detach(), acc.detach()

diabetes_ds = DiabetesDataset(csv_file, with_y=include_Y)
train_dataloader = DataLoader(diabetes_ds, batch_size=10, shuffle=True)
batches = len(train_dataloader)
loss = torch.zeros(batches)
acc = torch.zeros(batches)
for batch_idx, (features, labels) in enumerate(train_dataloader):
    print(f"{batch_idx}, {features.size()}, {labels.size()}")
    loss[batch_idx], acc[batch_idx] = iterate_batch(features, labels)



def render_accuracy_plot(batches, loss, acc):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(range(batches), loss);
    plt.title("CE loss");
    plt.xlabel("Batch Number");
    plt.subplot(1,2,2)
    plt.plot(range(batches), acc);
    plt.title("Accuracy");
    plt.xlabel("Batch Number");
    plt.show()


render_accuracy_plot(batches, loss, acc)

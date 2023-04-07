import torch
import sklearn.datasets as skds
import numpy as np
from pandas.core.array_algos import transforms
from torch.utils.data import Dataset, DataLoader
import os

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
x_df = pd.DataFrame(X, columns=bunch.feature_names)
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
    see https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """

    def __init__(self, file):
        """
        load the CSV to Panda frame
        """
        self._csv_df = pd.read_csv(file, sep='\t')
        # NOTE: Y is last column in CSV

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
        return self._transform(self._csv_df.iloc[idx, self._csv_df.columns != 'Y']),\
            self._csv_df.iloc[idx, self._csv_df.columns == 'Y'].item()


csv_file = os.path.join(os.path.dirname(__file__), "diabetes.csv")
if not os.path.exists(csv_file):
    raise EnvironmentError(f'{csv_file} not found')
diabetes_ds = DiabetesDataset(csv_file)
print(f"number of rows: {len(diabetes_ds)}")
feature, label = diabetes_ds[0]
print(f"first row: {feature}, {label}")
print(f"second row: {diabetes_ds[1]}")
print(f"third row: {diabetes_ds[2]}")
print(f"last row: {diabetes_ds[441]}")
print(f"last row: {diabetes_ds[-1]}")

ds_loader = DataLoader(diabetes_ds, batch_size=10, shuffle=False)

# iterable
ds_iter = iter(ds_loader)
print(f"batch_train_features: {ds_iter}")
batch_features, batch_labels = next(ds_iter)
print(batch_features, batch_labels)

for features, labels in ds_loader:
    print(f"{features}, {labels}")

enumerator = enumerate(ds_iter)
for batch_idx, (features, labels) in enumerator:
    print(f"{batch_idx}, {features}, {labels}")


#print(labels)

#
# def iterate_batch(imgs, labels):
#     """
#     see pae 59
#     :param imgs:
#     :param labels:
#     :return:
#     """
#     pass
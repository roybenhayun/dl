import torch
import sklearn.datasets as skds
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
from torch import nn

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


# 1D
csv_url = "https://iot-arch-bucket1.s3.eu-central-1.amazonaws.com/public/diabetes.csv"
Y = pd.read_csv(csv_url, sep='\t')
print("CSV loaded: ", Y)

decile_labels = [*range(1, 11)]
decile_labels.reverse()
y_df = pd.DataFrame(Y, columns=['Y'])
Y['Class'] = pd.qcut(y_df['Y'], 10, labels=decile_labels)
print(Y['Class'])
# assert Y['Class'].max() == 1, f"highest value {Y['Class'].max()} should be Class 1"
# assert Y['Class'].min() == 10, f"lowest value {Y['Class'].min()} should be Class 10"
min_idx_in_y = np.where(Y['Y'] == Y['Y'].min())
max_idx_in_y = np.where(Y['Y'] == Y['Y'].max())
print(f"at index {min_idx_in_y}, Value: {Y['Y'].iloc[min_idx_in_y].iloc[0]}, Class: {Y.iloc[min_idx_in_y].iloc[0]['Class']}")
print(f"at index {max_idx_in_y}, Value: {Y['Y'].iloc[max_idx_in_y].iloc[0]}, Class: {Y.iloc[max_idx_in_y].iloc[0]['Class']}")
assert Y.iloc[min_idx_in_y].iloc[0]['Class'] == 10, f"{Y.min()} should be Class 10 in Y"
assert Y.iloc[max_idx_in_y].iloc[0]['Class'] == 1, f"{Y.max()} should be Class 1 in Y"


# 1E


class DiabetesDataset(Dataset):
    """
    encapsulates the diabetes dataset
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
        if self._with_Y:
            return self._transform(self._csv_df.iloc[idx, self._csv_df.columns != 'Target']),\
                self._csv_df.iloc[idx, self._csv_df.columns == 'Target'].item()
        else:
            return self._transform(self._csv_df.iloc[idx, ~self._csv_df.columns.isin(['Y', 'Target'])]),\
                self._csv_df.iloc[idx, self._csv_df.columns == 'Target'].item()


csv_file = os.path.join(os.path.dirname(__file__), "diabetes.csv")  # NOTE: Y included in the CSV
if not os.path.exists(csv_file):
    raise EnvironmentError(f'{csv_file} not found')


def validate_diabetes_ds(ds):
    print("\n* validate_diabetes_ds")
    print(f"number of rows: {len(ds)}")
    assert len(ds) == 442, f"expected 442, got {len(ds)}"
    features, target = ds[0]
    print(f"first row: features: {features}, size: {features.size()}, target: {target}")
    features, target = ds[1]
    print(f"second row: features: {features}, size: {features.size()}, target: {target}")
    features, target = ds[2]
    print(f"third row: features: {features}, size: {features.size()}, target: {target}")
    features, target = ds[441]
    print(f"last row: features: {features}, size: {features.size()}, target: {target}")
    features, target = ds[-1]
    print(f"last row: features: {features}, size: {features.size()}, target: {target}")


validate_diabetes_ds(DiabetesDataset(csv_file, with_y=True))
validate_diabetes_ds(DiabetesDataset(csv_file, with_y=False))


# 1.6, 1.7

diabetes_ds = DiabetesDataset(csv_file, with_y=True)
ds_loader = DataLoader(diabetes_ds, batch_size=10, shuffle=False)

for features, labels in ds_loader:
    pass
    #print(f"{features}, {labels}")

ds_iter = iter(ds_loader)
enumerator = enumerate(ds_iter)
for batch_idx, (features, labels) in enumerator:
    pass
    #print(f"{batch_idx}, {features}, {labels}")

# iterable
ds_iter = iter(ds_loader)
print(f"batch_train_features: {ds_iter}")
batch_features, batch_labels = next(ds_iter)
print(batch_features, batch_labels)


# 1.8 - החוזה את Class על סמך שאר המשתנים




def iterate_batch(input_tensor, labels):
    optimizer.zero_grad()

    # forward pass
    y_model = model(input_tensor)

    # compute loss
    loss = CE_loss(y_model, labels.long() - 1)  # must accept long, and Target < model output_size. Deciles start from 1, so need to shift -1

    # backpropagation - backward pass
    loss.backward()

    # update network weights
    optimizer.step()

    # count predicted labels
    predicted_labels = y_model.argmax(dim=1)

    acc = (predicted_labels == labels).sum() / len(labels)
    return loss.detach(), acc.detach()


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

#
# with Y
#

include_Y = True
input_tensor_size = 11 if include_Y else 10  # the labels are the 11 deciles or 10 deciles w\wo Y respectively
output_size = 10  # number of Target labels
model = nn.Sequential(nn.Linear(input_tensor_size, output_size),  # input size, output size
                      nn.LogSoftmax(dim=1))  # activation on the output column
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
CE_loss = nn.NLLLoss()
print(model)

diabetes_ds_wY = DiabetesDataset(csv_file, with_y=include_Y)
train_dataloader = DataLoader(diabetes_ds_wY, batch_size=10, shuffle=True)
batches = len(train_dataloader)
loss = torch.zeros(batches)
acc = torch.zeros(batches)
for batch_idx, (features, labels) in enumerate(train_dataloader):
    print(f"{batch_idx}, {features.size()}, {labels.size()}")
    loss[batch_idx], acc[batch_idx] = iterate_batch(features, labels)

render_accuracy_plot(batches, loss, acc, "Diabetes DS: predict Class\deciles, 1 epoch, with Y")

#
# without Y
#


input_tensor_size = 10  # 10 wo Y
output_size = 10  # number of Target labels
model = nn.Sequential(nn.Linear(input_tensor_size, output_size), nn.LogSoftmax(dim=1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
CE_loss = nn.NLLLoss()
print(model)

diabetes_ds_woY = DiabetesDataset(csv_file, with_y=False)
train_dataloader = DataLoader(diabetes_ds_woY, batch_size=10, shuffle=True)
batches = len(train_dataloader)
loss = torch.zeros(batches)
acc = torch.zeros(batches)
for batch_idx, (features, labels) in enumerate(train_dataloader):
    print(f"{batch_idx}, {features.size()}, {labels.size()}")
    loss[batch_idx], acc[batch_idx] = iterate_batch(features, labels)

render_accuracy_plot(batches, loss, acc, "Diabetes DS, predict Class\deciles, 1 epoch, without Y")


# 1.10

# https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split


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


def get_test_batch_accuracy(input_tensor, labels):
    """
    similar operation as iterate_batch() without Gradient calculation and Optimizer operations, for test evaluation only
    """

    # forward pass
    y_model = model(input_tensor)

    # calculate loss - just for stats, not doing backpropagation.
    loss = CE_loss(y_model, labels.long() - 1)

    # count predicated labels
    predicted_labels = y_model.argmax(dim=1)

    total_acc = (predicted_labels == labels).sum()
    acc = total_acc / len(labels)
    return loss.detach(), acc.detach(), total_acc


def train_and_test_subsets(test_set_size, train_set_size, dataset, num_epochs=20):
    subsets = random_split(dataset, [train_set_size, test_set_size])
    print(f"train set: {len(subsets[0])}, test set: {len(subsets[1])}")

    # train model

    model.train()  # switch to training mode

    train_dataloader = DataLoader(subsets[0], batch_size=10, shuffle=True)
    train_set_loss = torch.zeros(len(train_dataloader) * num_epochs)
    train_set_acc = torch.zeros(len(train_dataloader) * num_epochs)
    print("run train set..")
    idx = 0
    for epoch in tqdm(range(num_epochs), unit="epoch"):
        for batch_idx, (features, labels) in enumerate(train_dataloader):
            train_set_loss[idx], train_set_acc[idx] = iterate_batch(features, labels)
            idx += 1
    print(f"avg loss: {round(float(np.average(train_set_loss)), 3)}, avg acc: {round(float(np.average(train_set_acc)), 3)}")

    # test model

    test_dataloader = DataLoader(subsets[1], batch_size=10, shuffle=True)
    test_set_loss = torch.zeros(len(test_dataloader))
    test_set_acc = torch.zeros(len(test_dataloader))
    total_acc = 0
    print("run test set..")
    model.eval()  # switch to evaluation mode
    with torch.no_grad():
        for batch_idx, (features, labels) in tqdm(enumerate(test_dataloader), unit="batch"):
            test_set_loss[batch_idx], test_set_acc[batch_idx], batch_acc = get_test_batch_accuracy(features, labels)
            total_acc += batch_acc

    print(f"avg loss: {round(float(np.average(test_set_loss)), 3)}, avg acc: {round(float(np.average(test_set_acc)), 3)}")

    return len(train_dataloader) * num_epochs, train_set_loss, train_set_acc, \
        len(test_dataloader), test_set_loss, test_set_acc


print("train and test, with Y...")
diabetes_ds_wY = DiabetesDataset(csv_file, with_y=True)
model = nn.Sequential(nn.Linear(11, 10), nn.LogSoftmax(dim=1))
CE_loss = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
train_set_size = int(len(diabetes_ds_wY) * 0.8)
test_set_size = len(diabetes_ds_wY) - train_set_size
num_epochs = 1
render_train_test_accuracy_plot(* train_and_test_subsets(test_set_size, train_set_size, diabetes_ds_wY, num_epochs=num_epochs),
                                f"Diabetes: predict decile, {num_epochs} epochs, with Y")

print("train and test, without Y...")
diabetes_ds_woY = DiabetesDataset(csv_file, with_y=False)
model = nn.Sequential(nn.Linear(10, 10), nn.LogSoftmax(dim=1))
CE_loss = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
train_set_size = int(len(diabetes_ds_woY) * 0.8)
test_set_size = len(diabetes_ds_woY) - train_set_size
render_train_test_accuracy_plot(* train_and_test_subsets(test_set_size, train_set_size, diabetes_ds_woY, num_epochs=num_epochs),
                                f"Diabetes: predict decile, {num_epochs} epochs, without Y")



#
# 1.11
#


class PercentageDiabetesDataset(DiabetesDataset):
    def __init__(self, file, with_y=True):
        super().__init__(file, with_y)
        # override 'Target' column
        print("before: ", self._csv_df['Target'])
        percent_labels = [*range(1, 101)]
        self._csv_df['Target'] = pd.qcut(pd.DataFrame(self._csv_df, columns=['Y'])['Y'], 100, percent_labels)
        print("after: ", self._csv_df['Target'])


diabetes_perc_ds_wY = PercentageDiabetesDataset(csv_file, with_y=True)
#validate_diabetes_ds(diabetes_perc_ds_wY)
diabetes_perc_ds_woY = PercentageDiabetesDataset(csv_file, with_y=False)
#validate_diabetes_ds(diabetes_perc_ds_woY)

# verify 25 is 100, 325 is 1
Y = diabetes_perc_ds_wY._csv_df['Y']
Target = diabetes_perc_ds_wY._csv_df['Target']
min_idx_in_y = np.where(Y == Y.min())
max_idx_in_y = np.where(Y == Y.max())
print(f"at index ({min_idx_in_y}), Value: ({Y.iloc[min_idx_in_y]}), Class: ({Target.iloc[min_idx_in_y]})")
print(f"at index ({max_idx_in_y}), Value: ({Y.iloc[max_idx_in_y]}), Class: ({Target.iloc[max_idx_in_y]})")

# assert Y.iloc[min_idx_in_y].iloc[0]['Class'] == 10, f"{Y.min()} should be Class 10 in Y"
# assert Y.iloc[max_idx_in_y].iloc[0]['Class'] == 1, f"{Y.max()} should be Class 1 in Y"

#
# 1.11 - replay 8-10
#


include_Y = True
input_tensor_size = 11 if include_Y else 10  # the labels are the 11 deciles or 10 deciles w\wo Y respectively
output_size = 100  # number of Target labels - 100 percent
model = nn.Sequential(nn.Linear(input_tensor_size, output_size),  nn.LogSoftmax(dim=1))
CE_loss = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

train_dataloader = DataLoader(diabetes_perc_ds_wY, batch_size=10, shuffle=True)
batches = len(train_dataloader)
loss = torch.zeros(batches)
acc = torch.zeros(batches)
for batch_idx, (features, labels) in enumerate(train_dataloader):
    print(f"{batch_idx}, {features.size()}, {labels.size()}")
    loss[batch_idx], acc[batch_idx] = iterate_batch(features, labels)

render_accuracy_plot(batches, loss, acc, "Percentage-Diabetes DS, predict percentage with Y")


model = nn.Sequential(nn.Linear(10, 100),  nn.LogSoftmax(dim=1))
CE_loss = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

train_dataloader = DataLoader(diabetes_perc_ds_woY, batch_size=10, shuffle=True)
batches = len(train_dataloader)
loss = torch.zeros(batches)
acc = torch.zeros(batches)
for batch_idx, (features, labels) in enumerate(train_dataloader):
    print(f"{batch_idx}, {features.size()}, {labels.size()}")
    loss[batch_idx], acc[batch_idx] = iterate_batch(features, labels)

render_accuracy_plot(batches, loss, acc, "Percentage-Diabetes DS, predict percentage without Y")

# repeat 1.10 with train\test sets

print("train and test, with Y...")
model = nn.Sequential(nn.Linear(11, 100),  nn.LogSoftmax(dim=1))
CE_loss = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
train_set_size = int(len(diabetes_perc_ds_wY) * 0.8)
test_set_size = len(diabetes_perc_ds_wY) - train_set_size
num_epochs = 1
render_train_test_accuracy_plot(* train_and_test_subsets(test_set_size, train_set_size, diabetes_perc_ds_wY, num_epochs=num_epochs),
                                f"Diabetes: predict percentages, {num_epochs} epochs, with Y")

print("train and test, without Y...")
model = nn.Sequential(nn.Linear(10, 100),  nn.LogSoftmax(dim=1))
CE_loss = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
render_train_test_accuracy_plot(* train_and_test_subsets(test_set_size, train_set_size, diabetes_perc_ds_woY, num_epochs=num_epochs),
                                f"Diabetes: predict percentages, {num_epochs} epochs, with Y")

# 1.14: percentage is too fine-grained accuracy. Deciles is a better option, specific for this scenario and likely to other scenarios
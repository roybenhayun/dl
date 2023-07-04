import torch
from torch import nn
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from maman14.DropNorm import DropNorm


def iterate_batch(input_tensor, labels, model, optimizer, ce_loss):
    if model.training:
        optimizer.zero_grad()

    # forward pass
    y_model = model(input_tensor)

    # compute loss
    loss = ce_loss(y_model, labels.long())  # must accept long

    if model.training:
        # backpropagation - backward pass
        loss.backward()

        # update network weights
        optimizer.step()

    # count predicted labels
    predicted_labels = y_model.argmax(dim=1)

    total_predicted = (predicted_labels == labels).sum()
    acc = total_predicted / len(labels)
    return loss.detach(), acc.detach(), total_predicted


def render_accuracy_plot(unit, results, loss, acc, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(results), loss)
    plt.title("CE loss")
    plt.xlabel(f"{unit} Number")
    plt.subplot(1, 2, 2)
    plt.plot(range(results), acc)
    avg_acc = round(float(np.average(acc)), 3)
    plt.title(f"Accuracy (avg: {avg_acc})")
    plt.xlabel(f"{unit} Number")
    plt.suptitle(title)
    plt.show()


def train_and_test_fashion_mnist_nn(num_epochs=1):
    train_data_transformed = torchvision.datasets.FashionMNIST(root=r"C:\work_openu\DL\temp\fashion-mnist", train=True,
                                                               download=False,
                                                               transform=torchvision.transforms.PILToTensor())

    batch_size = 100
    train_dataloader = DataLoader(train_data_transformed, batch_size, shuffle=True)
    print(f"samples num: {len(train_data_transformed)}")
    print(f"classes num: {len(train_data_transformed.classes)}")
    batches = len(train_dataloader)
    print(f"batche size: {batch_size}")
    print(f"batches num: {len(train_dataloader)}")

    model = nn.Sequential(nn.Flatten(),  # flatten dimensions with size 1
                          nn.Linear(784, 100), nn.ReLU(),
                          nn.Linear(100, 10), nn.ReLU(),
                          nn.LogSoftmax(dim=1))
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 10),
        nn.LogSoftmax(dim=1)  # log(y)
    )
    print(f"model: {model}")
    module_names = '>'.join([type(module).__name__ for name, module in model.named_modules()])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ce_loss = nn.NLLLoss()

    print(f"epochs num: {num_epochs}")
    batch_loss = torch.zeros(num_epochs, batches)
    batch_acc = torch.zeros(num_epochs, batches)
    train_set_loss = torch.zeros(num_epochs)
    train_set_acc = torch.zeros(num_epochs)

    # Page 116, Q 1, 2
    #image_mask = torch.rand(28, 28) > 0.5

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        for batch_idx, (features, labels) in tqdm(enumerate(train_dataloader)):
            features = features.type(torch.float)  # change type to float
            #features = features * image_mask
            batch_loss[epoch, batch_idx], batch_acc[epoch, batch_idx], _ = iterate_batch(features, labels, model,
                                                                                         optimizer, ce_loss)
        train_set_loss[epoch] = float(np.average(batch_loss))
        train_set_acc[epoch] = float(np.average(batch_acc))
        print(f"epoch avg loss: {round(float(np.average(train_set_loss[epoch])), 3)}, "
              f"epoch avg acc: {round(float(np.average(train_set_acc[epoch])), 3)}")

    print(f"total avg loss: {round(float(np.average(train_set_loss)), 3)}, "
          f"total avg acc: {round(float(np.average(train_set_acc)), 3)}")

    render_accuracy_plot("Batch", batches, batch_loss[0, :], batch_acc[0, :],
                         f"Fashion-MNIST: *1st* Epoch per batch (batch size: {batch_size}")
    render_accuracy_plot("Epoch", num_epochs, train_set_loss, train_set_acc,
                         f"Fashion-MNIST: {num_epochs} epochs (batch size: {batch_size}")

    #
    # eval with test data
    #

    test_data_transformed = torchvision.datasets.FashionMNIST(root=r"C:\work_openu\DL\temp\fashion-mnist", train=False,
                                                              download=False,
                                                              transform=torchvision.transforms.PILToTensor())
    test_dataloader = DataLoader(test_data_transformed, batch_size, shuffle=True)
    samples_num = len(test_data_transformed)
    print(f"samples num: {samples_num}")
    print(f"batche size: {batch_size}")
    test_batches = len(test_dataloader)
    print(f"batches num: {test_batches}")

    model.eval()  # switch to evaluation mode
    test_set_loss = torch.zeros(len(test_dataloader))
    test_set_acc = torch.zeros(len(test_dataloader))
    total_acc = 0
    with torch.no_grad():
        for batch_idx, (features, labels) in tqdm(enumerate(test_dataloader), unit="batch"):
            # change type to float (needed in forward pass)
            features = features.type(torch.float)

            # features = features * image_mask

            test_set_loss[batch_idx], test_set_acc[batch_idx], predicted = iterate_batch(features, labels, model, optimizer, ce_loss)
            total_acc += predicted

    print(f"avg loss: {round(float(np.average(test_set_loss)), 3)}")
    print(f"avg acc: {round(float(np.average(test_set_acc)), 3)}")
    print(f"total accuracy: {total_acc} / {samples_num} = {round(total_acc.item() / samples_num, 3)}")
    render_accuracy_plot("Batch", test_batches, test_set_loss, test_set_acc,
                         f"Fashion-MNIST test set (ACC: {round(total_acc.item() / samples_num, 3)}")


if __name__ == '__main__':
    print("Fashion-MNIST plain vanilla")
    train_and_test_fashion_mnist_nn(num_epochs=10)


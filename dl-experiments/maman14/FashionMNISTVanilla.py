import torch
from torch import nn
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def iterate_batch(input_tensor, labels, model, optimizer, ce_loss):
    optimizer.zero_grad()

    # forward pass
    y_model = model(input_tensor)

    # compute loss
    loss = ce_loss(y_model, labels.long())  # must accept long

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


def train_fashion_mnist_nn():
    train_data_transformed = torchvision.datasets.FashionMNIST(root=r"C:\work_openu\DL\temp\fashion-mnist", train=True,
                                                               download=False,
                                                               transform=torchvision.transforms.PILToTensor())

    batch_size = 100
    train_dataloader = DataLoader(train_data_transformed, batch_size, shuffle=False)
    print(f"samples num: {len(train_data_transformed)}")
    print(f"classes num: {len(train_data_transformed.classes)}")
    batches = len(train_dataloader)
    print(f"batche size: {batch_size}")
    print(f"batches num: {len(train_dataloader)}")

    model = nn.Sequential(nn.Flatten(),
                          nn.Linear(784, 10),
                          nn.LogSoftmax(dim=1))
    print(f"model: {model}")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ce_loss = nn.NLLLoss()
    loss = torch.zeros(batches)
    acc = torch.zeros(batches)
    for batch_idx, (features, labels) in enumerate(train_dataloader):
        print(f"{batch_idx}, {features.size()}, {labels.size()}")
        # change type to float
        features = features.type(torch.float)
        loss[batch_idx], acc[batch_idx] = iterate_batch(features, labels, model, optimizer, ce_loss)

    render_accuracy_plot(batches, loss, acc, f"Fashion-MNIST: batch size: {batch_size}, network: [Linear] -> [LogSoftmax]")


if __name__ == '__main__':
    print("Fashion-MNIST plain vanilla")
    train_fashion_mnist_nn()


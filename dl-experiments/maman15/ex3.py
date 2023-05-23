import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from tqdm import tqdm



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


def train_and_test_cifar10(model, optimizer, transforms, single_batch=True):
    train_data_transformed = torchvision.datasets.CIFAR10(root=r"C:\work_openu\DL\temp\cifar-10",
                                                          train=True, download=False,
                                                          transform=transforms)

    batch_size = 100
    train_dataloader = DataLoader(train_data_transformed, batch_size, shuffle=True)
    print(f"samples num: {len(train_data_transformed)}")
    print(f"classes num: {len(train_data_transformed.classes)}")
    batches = len(train_dataloader)
    print(f"batche size: {batch_size}")
    print(f"batches num: {len(train_dataloader)}")

    ce_loss = nn.NLLLoss()
    if single_batch:  # actually, it's a single epoch. it's the chart that is per batches..
        loss = torch.zeros(batches)
        acc = torch.zeros(batches)
        for batch_idx, (features, labels) in enumerate(train_dataloader):
            print(f"{batch_idx}, {features.size()}, {labels.size()}")
            # change type to float (needed in forward pass)
            features = features.type(torch.float)
            loss[batch_idx], acc[batch_idx], _ = iterate_batch(features, labels, model, optimizer, ce_loss)
            print(f"loss: {loss[batch_idx]}, acc: {acc[batch_idx]}")

        render_accuracy_plot("Batch", batches, loss, acc,
                             f"ResNet/CIFAR10 single-batch (batch size: {batch_size}, nn: ResNet18/CIFAR10 Classifier)")
    else:
        num_epochs = 10
        print(f"epochs num: {num_epochs}")
        batch_loss = torch.zeros(batches)
        batch_acc = torch.zeros(batches)
        train_set_loss = torch.zeros(num_epochs)
        train_set_acc = torch.zeros(num_epochs)

        for epoch in tqdm(range(num_epochs), unit="epoch"):
            for batch_idx, (features, labels) in enumerate(train_dataloader):
                features = features.type(torch.float)  # change type to float
                batch_loss, batch_acc, _ = iterate_batch(features, labels, model, optimizer, ce_loss)
            train_set_loss[epoch] = float(np.average(batch_loss))
            train_set_acc[epoch] = float(np.average(batch_acc))
            print(f"epoch avg loss: {round(float(np.average(train_set_loss[epoch])), 3)}, "
                  f"epoch avg acc: {round(float(np.average(train_set_acc[epoch])), 3)}")

        print(f"total avg loss: {round(float(np.average(train_set_loss)), 3)}, "
              f"total avg acc: {round(float(np.average(train_set_acc)), 3)}")

        render_accuracy_plot("Epoch", num_epochs, train_set_loss, train_set_acc,
                             f"ResNet/CIFAR10 {num_epochs} epochs (batch size: {batch_size}, nn: ResNet18/CIFAR10 Classifier)")

    #
    # eval with test data
    #

    test_data_transformed = torchvision.datasets.FashionCIFAR10MNIST(root=r"C:\work_openu\DL\temp\cifar-10",
                                                                     train=False, download=False,
                                                                     transform=transforms)
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
            test_set_loss[batch_idx], test_set_acc[batch_idx], predicted = iterate_batch(features, labels, model, optimizer, ce_loss)
            total_acc += predicted

    print(f"avg loss: {round(float(np.average(test_set_loss)), 3)}")
    print(f"avg acc: {round(float(np.average(test_set_acc)), 3)}")
    print(f"total accuracy: {total_acc} / {samples_num} = {round(total_acc.item() / samples_num, 3)}")
    render_accuracy_plot("Batch", test_batches, test_set_loss, test_set_acc,
                         f"CIFAR10 test set (ACC: {round(total_acc.item() / samples_num, 3)}, nn: {module_names})")


if __name__ == '__main__':
    print("ex- main")

    # 3.A
    train_data_transformed = torchvision.datasets.CIFAR10(root=r"C:\work_openu\DL\temp\cifar-10",
                                                          train=False, download=True,
                                                          transform=torchvision.transforms.PILToTensor())

    print(train_data_transformed)

    cifar10_batch_size = 10
    train_dataloader = DataLoader(train_data_transformed, cifar10_batch_size, shuffle=True)
    print(f"samples num: {len(train_data_transformed)}")
    cifar10_classes = train_data_transformed.classes
    print(f"classes num: {len(cifar10_classes)}")
    cifar10_batches = len(train_dataloader)
    print(f"batche size: {cifar10_batch_size}")
    print(f"batches num: {len(train_dataloader)}")

    # 3.C
    iterator = iter(train_dataloader)
    cifar10_imgs, cifar10_labels = next(iterator)
    class_names = train_data_transformed.classes
    fig = plt.figure()
    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1)
        # plt.imshow requires the color dimension last. need to permute from [3, M, N] to [M, N, 3]
        transposed_img = cifar10_imgs[i].permute(1, 2, 0)
        plt.imshow(transposed_img)
        ax.set_title(class_names[cifar10_labels[i]])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    plt.show()

    # 3.D

    # resnet18 = models.resnet18(pretrained=True)  # deprecated. see https://pytorch.org/vision/stable/models.html
    # Before using the pre-trained models, one must preprocess the image
    # (resize with right resolution/interpolation, apply inference transforms, rescale the values etc).
    # All the necessary information for the inference transforms of each pre-trained model is provided on
    # its weights documentation

    resnet18_weights = models.ResNet18_Weights.IMAGENET1K_V1
    print(f"ResNet weights default: {models.ResNet18_Weights.DEFAULT}")
    print(f"using: {resnet18_weights}")
    resnet18_preprocess_transforms = resnet18_weights.transforms()
    print(f"ResNet transforms: {resnet18_preprocess_transforms}")

    resnet18 = models.resnet18(weights=resnet18_weights)
    print("----------------------")
    print("ResNet model:\n")
    print(resnet18)
    # for name, module in resnet18.named_children():
    #     print(f"{name}, {module}")
    print("=============================")
    print("model parameters:\n")
    for param in resnet18.parameters():
        print(type(param), param.size())
    print("----------------------")

    last_layers = list(resnet18.children())[-2:]
    print("model last 2 layers:\n")
    print(*last_layers, sep="\n")
    print("----------------------")

    # 3.E
    # see p-148
    # resnet18 Classifer: (fc): Linear(in_features=512, out_features=1000, bias=True)
    cifar10_classifier = Linear(in_features=512, out_features=len(cifar10_classes), bias=True)
    print(f"new classifier:\n {cifar10_classifier}")

    #
    # 3.F
    #

    # 3.F.1 - transforms
    cifar10_to_resnet18_transforms = T.Compose([
        T.ToTensor(),  # convert from [0, 255] to float [0, 1]
        T.Resize(224),
        T.Normalize(resnet18_preprocess_transforms.mean, resnet18_preprocess_transforms.std)])

    # 3.F.2 - optimize only Classifier
    cifar10_optimizer = torch.optim.SGD(cifar10_classifier.parameters(), lr=0.1)

    # 3.F.3 - disable autograd
    for param in resnet18.parameters():
        param.requires_grad = False

    #
    # 3.F + 3.G : train cifar10_classifier on CIFAR-10, test on test set until reaching 70%
    #

    resnet18.fc = cifar10_classifier
    print(f"modified ResNet FC classifier: {resnet18.fc}")
    for param in resnet18.parameters():
        print(f"ResNet layer requires_grad: {param.requires_grad}")

    train_and_test_cifar10(resnet18, cifar10_optimizer, cifar10_to_resnet18_transforms, True)

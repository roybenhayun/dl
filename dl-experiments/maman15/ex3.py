import torch
from torch.nn import Linear
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as T

# 3.A
train_data_transformed = torchvision.datasets.CIFAR10(root=r"C:\work_openu\DL\temp\cifar-10", train=False,
                                                           download=True,
                                                           transform=torchvision.transforms.PILToTensor())

print(train_data_transformed)

batch_size = 10
train_dataloader = DataLoader(train_data_transformed, batch_size, shuffle=True)
print(f"samples num: {len(train_data_transformed)}")
cifar10_classes = train_data_transformed.classes
print(f"classes num: {len(cifar10_classes)}")
batches = len(train_dataloader)
print(f"batche size: {batch_size}")
print(f"batches num: {len(train_dataloader)}")

# 3.C
iterator = iter(train_dataloader)
imgs, labels = next(iterator)
class_names = train_data_transformed.classes
fig = plt.figure()
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    # plt.imshow requires the color dimension last. need to permute from [3, M, N] to [M, N, 3]
    transposed_img = imgs[i].permute(1, 2, 0)
    plt.imshow(transposed_img)
    ax.set_title(class_names[labels[i]])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

plt.show()

# 3.D

#resnet18 = models.resnet18(pretrained=True)  # deprecated. see https://pytorch.org/vision/stable/models.html
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
optimizer = torch.optim.SGD(cifar10_classifier.parameters(), lr=0.1)

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

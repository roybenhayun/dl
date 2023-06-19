# -*- coding: utf-8 -*-
"""22961_7_2_MNIST_autoencoder_and_denoising_AE - MAMAN17 - ex1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ayFFoKSwPlmPXqGXBTGXMmRX_K0Bbazb

# Train FC autoencoder
"""

import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
# TODO: replace FashionMNIST with MNIST
train_data_transformed = torchvision.datasets.MNIST(
    root="/22961", train=True, download=True,
    transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(
    train_data_transformed, batch_size=1024)

img, _ = next(iter(train_dataloader))

test_data_transformed = torchvision.datasets.MNIST(
    root="/22961", train=False, download=True,
    transform=torchvision.transforms.ToTensor())
test_dataloader = DataLoader(
    test_data_transformed, batch_size=1024)

if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')
print(device)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.linear_f1 = nn.Linear(784,latent_dim)
        self.linear_f2 = nn.Linear(latent_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, image):
      flattned = image.flatten(start_dim=1)
      Ae = self.linear_f1(flattned)
      self.Ae = self.relu(Ae)
      self.Be = self.linear_f2(self.Ae)
      self.compressed_image = self.relu(self.Be)
      return self.compressed_image

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.linear_f2I = nn.Linear(latent_dim, latent_dim) # should be inverse function of linear_f2
        self.relu = nn.ReLU()
        self.linear_f1I = nn.Linear(latent_dim, 784)  # should be inverse function of linear_f1
        self.sigmoid = nn.Sigmoid()

    def forward(self, compressed_image):
      self.Bd = self.linear_f2I(compressed_image)
      Bd = self.relu(self.Bd)
      Ad = self.linear_f1I(Bd)
      decoded = self.sigmoid(Ad)
      reconstructed_image = decoded.reshape(-1,1,28,28)
      return reconstructed_image


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2)
        self.relu = nn.ReLU()
        self.f1 = self.conv1

    def forward(self, image):
      self.Ae = self.conv1(image)   # torch.Size([1024, 32, 28, 28]) - note: can't keep the result here for MSE comparison as it already passed misc operations in the Encoder
      temp = self.relu(self.Ae)     # relu(Ae)
      return temp

class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1I = nn.ConvTranspose2d(32, 1, 4, stride=2)  # TODO: kernel size 1 is meaningless?
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.f1I = self.conv1I

    def forward(self, feature_map):
      self.Ad = self.conv1I(feature_map)           # torch.Size([1024, 1, 28, 28])
      reconstructed_image = self.sigmoid(self.Ad)  # sigmoid(Ad)
      return reconstructed_image  # TODO: note the return must be [1024, 1, 28, 28], same as the imgs_batch


latent_dim  = 10
autoencoder = nn.Sequential(ConvEncoder(), ConvDecoder()).to(device)
optimizer   = torch.optim.AdamW(autoencoder.parameters())
MSELoss = nn.MSELoss()  # MSE measures proximity between results


def inverse_conv_mapping_loss(autoencoder, imgs_batch):
    inverse_f1_out = autoencoder[1].f1I(autoencoder[0].f1(imgs_batch))  # f1I should undo f1
    inverse_f1_loss = MSELoss(imgs_batch, inverse_f1_out)  # so imgs_batch and inverse_f1_out should be close
    return inverse_f1_loss  # TODO: add loss for two layers, not just one


def inverse_mapping_loss(autoencoder, imgs_batch):
    flattned = imgs_batch.flatten(start_dim=1)
    inverse_f1_out = autoencoder[1].linear_f1I(autoencoder[0].linear_f1(flattned))  # linear_f1I should undo linear_f1
    inverse_f1_loss = MSELoss(flattned, inverse_f1_out)  # so flattned and inverse_f1_out should be close
    inverse_f2_out = autoencoder[1].linear_f2I(autoencoder[0].linear_f2(autoencoder[0].Ae))  # linear_f2I should undo linear_f2
    inverse_f2_loss = MSELoss(autoencoder[0].Ae, inverse_f2_out)  # so flattned and inverse_f2_out should be close
    return inverse_f1_loss + inverse_f2_loss  # TODO: add loss for two layers, not just one

def iterate_batch(imgs):
  imgs = imgs.to(device)
  optimizer.zero_grad()
  reconstructed = autoencoder(imgs)
  reconstructed_loss = MSELoss(reconstructed, imgs)
  #inverse_loss = inverse_mapping_loss(autoencoder, imgs)
  inverse_loss = inverse_conv_mapping_loss(autoencoder, imgs)
  loss = reconstructed_loss + (0.001 * inverse_loss)  # inverse_loss added, tuned.
  loss.backward()
  optimizer.step()
  return loss, inverse_loss

batches=len(train_dataloader)
epochs = 12
batch_loss = torch.empty(batches, device=device)
batch_inverse_loss = torch.empty(batches, device=device)
epoch_loss =torch.empty(epochs, device=device)
epoch_inverse_loss =torch.empty(epochs, device=device)
for epoch_idx in tqdm(range(epochs)):
  for batch_idx, (imgs, _) in enumerate(train_dataloader):
    batch_loss[batch_idx], batch_inverse_loss[batch_idx] = iterate_batch(imgs)
  with torch.no_grad():
    epoch_loss[epoch_idx] = batch_loss.mean()
    epoch_inverse_loss[epoch_idx] = batch_inverse_loss.mean()

"""# display training results"""

print(f"epoch_loss[epoch_idx]: {epoch_loss[epoch_idx]}")
print(f"epoch_inverse_loss[epoch_idx]: {epoch_inverse_loss[epoch_idx]}")

plt.title("Epoch loss")
plt.plot(epoch_loss[:epoch_idx+1].cpu().detach());
plt.plot(epoch_inverse_loss[:epoch_idx+1].cpu().detach());
plt.show()

plt.title("Batch loss")
plt.plot(batch_loss.cpu().detach());
plt.plot(batch_inverse_loss.cpu().detach());
plt.show()



imgs, labels = next(iter(test_dataloader))

num_images = 10
with torch.no_grad():
  reconstructed = autoencoder(imgs.to(device)).cpu()
  fig,axes = plt.subplots(2, num_images, sharey=True)
  fig.set_figheight(4)
  fig.set_figwidth(20)
  rand_idx = torch.randint(size=(num_images,), high=imgs.size(0))
  for idx in range(num_images):
    axes[0,idx].imshow(imgs[rand_idx[idx],...].reshape(28,28).detach(), cmap='Greys')
    axes[0,idx].axes.get_xaxis().set_visible(False)
    axes[0,idx].axes.get_yaxis().set_visible(False)


    axes[1,idx].imshow(reconstructed[[rand_idx[idx]],...].reshape(28,28).detach(), cmap='Greys')
    axes[1,idx].axes.get_xaxis().set_visible(False)
    axes[1,idx].axes.get_yaxis().set_visible(False)

plt.show()

num_images=2
with torch.no_grad():
  reconstructed = autoencoder(imgs.to(device)).cpu()
  fig,axes = plt.subplots(num_images,2)
  fig.set_figheight(10)
  fig.set_figwidth(9)
  for idx in range(num_images):
    axes[idx,0].imshow(imgs[idx,...].reshape(28,28).detach(), cmap='Greys')
    axes[idx,0].axes.get_xaxis().set_visible(False)
    axes[idx,0].axes.get_yaxis().set_visible(False)


    axes[idx,1].imshow(reconstructed[idx,...].reshape(28,28).detach(), cmap='Greys')
    axes[idx,1].axes.get_xaxis().set_visible(False)
    axes[idx,1].axes.get_yaxis().set_visible(False)
axes[0,0].set_title("Original");
axes[0,1].set_title("Reconstructed");

plt.show()

"""# Compare intermediate products and F(x) * F(x)^(-1) = X"""

# print(f"Encoder weights shape:")
# print(f"linear_f1.weight.shape: {autoencoder[0].linear_f1.weight.shape}, Encoder.Ae.shape: {autoencoder[0].Ae.shape}")
# print(f"linear_f2.weight.shape: {autoencoder[0].linear_f2.weight.shape}, Encoder.Be.shape: {autoencoder[0].Be.shape}")
# print("Decoder weights shape:")
# print(f"linear_f2I.weight.shape: {autoencoder[1].linear_f2I.weight.shape}, Decoder.Bd.shape: {autoencoder[1].Bd.shape}")
# print(f"linear_f1I.weight.shape: {autoencoder[1].linear_f1I.weight.shape}, Decoder.Ad.shape: {autoencoder[1].Ad.shape}")

# print("Encoder weights:")
# print(f"linear_f1.weight: {autoencoder[0].linear_f1.weight}")
# print(f"linear_f2.weight: {autoencoder[0].linear_f2.weight}")
# print("Decoder weights:")
# print(f"linear_f2I.weight: {autoencoder[1].linear_f2I.weight}")
# print(f"linear_f1I.weight: {autoencoder[1].linear_f1I.weight}")

#
# compare expected result of f(x)*f(x)^(-1) == x
#

# TODO: maybe need to do in Eval mode

imgs, labels = next(iter(test_dataloader))


num_images=2
with torch.no_grad():
  Ae = autoencoder[0](imgs)  # Encoder.forward() expects Tensor [1024,1,28,28]
  reconstructed = autoencoder[1](Ae)  # Decoder.forward() expects Tensor [1024,10], returns Tensor [1024,1,28,28]
  fig,axes = plt.subplots(num_images,4)
  fig.set_figheight(10)
  fig.set_figwidth(9)
  for idx in range(num_images):
    axes[idx,0].imshow(imgs[idx,...].reshape(28,28).detach(), cmap='Greys')
    axes[idx,0].axes.get_xaxis().set_visible(False)
    axes[idx,0].axes.get_yaxis().set_visible(False)
    axes[idx,1].imshow(reconstructed[idx,...].reshape(28,28).detach(), cmap='Greys')
    axes[idx,1].axes.get_xaxis().set_visible(False)
    axes[idx,1].axes.get_yaxis().set_visible(False)

axes[0,0].imshow(Ae.reshape(28,28).detach(), cmap='Greys')
axes[0,0].axes.get_xaxis().set_visible(False)
axes[0,0].axes.get_yaxis().set_visible(False)
axes[0,1].imshow(reconstructed.reshape(28,28).detach(), cmap='Greys')
axes[0,1].axes.get_xaxis().set_visible(False)
axes[0,1].axes.get_yaxis().set_visible(False)

axes[0,0].set_title("Original");
axes[0,1].set_title("Reconstructed");


with torch.no_grad():
  Encoder_Ae = autoencoder[0].Ae
  Encoder_Be = autoencoder[0].Be
  print(f"Encoder_Ae: {Encoder_Ae}")
  print(f"Encoder_Be: {Encoder_Be}")
  Decoder_Ad = autoencoder[1].Ad
  Decoder_Bd = autoencoder[1].Bd
  print(f"Decoder_Ad: {Decoder_Ad}")
  print(f"Decoder_Bd: {Decoder_Bd}")

# print("Be vs Bd")

# # TODO: re-calculation with the layers - overkill?

# print("A1 vs A2 * linear_f1 * linear_f1I")

# print("B1 vs B2 * linear_f2 * linear_f2I")

# TODO: will this comparison give any indication??
print("A1 vs A2")
print("B1 vs B2")

"""# Same with Fully Connected Conv
NOTE: need to work with the Kernel params
"""
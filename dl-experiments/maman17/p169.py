import torch
from torch import nn
import sklearn.datasets as skds
import matplotlib.pyplot as plt
X, Y = skds.make_blobs(n_samples=100, n_features=2,
                       centers=2, random_state=1)
X = torch.tensor(X).float()
Y = torch.tensor(Y).long()
X = (X - X.mean(dim=0)) / X.std(dim=0)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="Greys", edgecolor="black");
plt.title("X distribution")
plt.show()

encoder = nn.Linear(2,1)
decoder = nn.Linear(1,2)
autoencoder = nn.Sequential(encoder, decoder)

optimizer = torch.optim.SGD(autoencoder.parameters(), lr=0.1)
MSELoss   = nn.MSELoss()

def iterate_epoch():
  optimizer.zero_grad()
  reconstructed_X = autoencoder(X)
  loss = MSELoss(reconstructed_X, X)
  loss.backward()
  optimizer.step()
  return loss

epochs = 40
epoch_loss = torch.empty(epochs)
for epoch_idx in range(epochs):
    epoch_loss[epoch_idx] = iterate_epoch().detach()

plt.title("reconstructed X loss")
plt.plot(epoch_loss);
plt.show()


class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_extractor_output):
        class_scores = self.linear(feature_extractor_output)
        probs = self.sigmoid(class_scores)
        return probs


class LatentEncodingClassifier(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, X):
        latent_encoding = encoder(X)
        predictions = classifier_head(latent_encoding)
        return predictions


classifier_head = ClassificationHead()  # single feature
classifying_encoder = LatentEncodingClassifier(encoder, classifier_head)

optimizer = torch.optim.SGD(classifier_head.parameters(), lr=0.1)
ce_loss = nn.NLLLoss()
loss = torch.zeros(epochs)

for epoch_idx in range(epochs):
    optimizer.zero_grad()
    predictions = classifying_encoder(X)
    loss = ce_loss(predictions.view(100), Y)
    loss.backward()
    epoch_loss[epoch_idx] = loss.detach()
    count_gt = (predictions > 0.5).sum().item()
    count_leq = (predictions <= 0.5).sum().item()
    print(f"{epoch_idx} > count_gt: {count_gt}, count_leq: {count_leq}")
    optimizer.step()

module_names = '>'.join([type(module).__name__ for name, module in classifying_encoder.named_modules()])
plt.suptitle("train classifier head")
plt.title(module_names)
plt.plot(epoch_loss);
plt.show()

with torch.no_grad():
  predictions_X = classifying_encoder(X)
print(predictions_X.size())
predictions_dist = torch.where(predictions_X > 0.5, 0, 1)
count_gt = (predictions_X > 0.5).sum().item()
count_leq = (predictions_X <= 0.5).sum().item()
print(f"count_gt: {count_gt}, count_leq: {count_leq}")


with torch.no_grad():
  reconstructed_X = autoencoder(X)
  encoded_X = encoder(X)
print(encoded_X.size())

fig,axes = plt.subplots(1,2)
fig.set_figheight(8)
fig.set_figwidth(18)

axes[0].scatter(reconstructed_X[:, 0], reconstructed_X[:, 1],
              c=Y, cmap="Greys", edgecolor="red", alpha=1);
axes[0].scatter(X[:, 0], X[:, 1],
              c=Y, cmap="Greys", edgecolor="black", alpha=0.3);
axes[0].set_title("Original and Reconstructed Data", size=20);

axes[1].scatter(encoded_X[:, 0], torch.zeros(encoded_X.size()),
              c=Y, cmap="Greys", edgecolor="black", alpha=1);
axes[1].axes.get_yaxis().set_visible(False)
axes[1].set_title("Latent Space", size=20);

plt.show()


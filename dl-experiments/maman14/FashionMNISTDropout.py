import torch
from torch import nn

class Dropout(nn.Module):
    def __init__(self, drop_rate=0.5):
        super().__init__()
        assert (0 < drop_rate < 1)
        self.drop_rate = drop_rate

    def forward(self, X):
        if self.training:
            mask = torch.rand(X.shape) > self.drop_rate
            return X * mask
        else:
            return (1 - self.drop_rate) * X


def iterate_batch(imgs, labels):
    pass

if __name__ == '__main__':
    print("ex 2-D")
    model_dropout = nn.Sequential(nn.Flatten(),
                                  nn.Linear(784, 100), nn.ReLU(),
                                  Dropout(),
                                  nn.Linear(100, 10), nn.ReLU(),
                                  Dropout(),
                                  nn.Linear(10, 10),
                                  nn.LogSoftmax(dim=1))

    CE_loss = nn.NLLLoss()
    optimizer = torch.optim.SGD(model_dropout.parameters(), lr=0.1)


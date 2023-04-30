import torch
from torch import nn


class DropNorm(nn.Module):
    def __init__(self):
        super().__init__()

        print("---------------------------------")
        print("* init module")
        print("DropNorm:")
        print("---------------------------------")

    def forward(self, batch_input):
        print("---------------------------------")
        Y = None
        print("* forward pass")
        mask = None
        batch_stddev = -1
        batch_variance = -1
        epsilon = 0.000001
        gamma = -1
        beta = -1
        print(f"forward pass result: {Y.shape}")
        print("---------------------------------")
        return Y


if __name__ == '__main__':
    print("ex 2")

    batch_size = 10
    features_num = 4 * 2  # ensure even number (that can split more than once)
    N = batch_size
    M = features_num

    d1 = DropNorm(batch_size, features_num)
    t1 = torch.ones(size=(batch_size, features_num))
    ret = d1(t1)


import math

import torch
from torch import nn
import numpy as np


class DropNorm(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        print(f"* init DropNorm p:{p}")
        self._p = p
        self.gamma = 1
        self.beta = 0
        # declare as network parameters so will be learnt (p87)

    def forward(self, x):
        # print("---------------------------------")
        # print(f"+ DropNorm.forward({batch_input.shape})")
        global gamma
        gamma = nn.Parameter(torch.ones(x.shape))
        global beta
        beta = nn.Parameter(torch.zeros(x.shape))


        # create binary mask with half random elements as zeros
        rand_indices = torch.randn(x.shape)
        mask_flat = torch.reshape(torch.ones(size=rand_indices.shape), (-1,))  # flatten
        # generate a random sample from indices [0 .. mask_flat.numel()]
        zero_indices = np.random.choice(mask_flat.numel(), int(rand_indices.numel() * self._p), replace=False)
        mask_flat[zero_indices] = 0  # create the 1\0 mask
        mask = torch.reshape(mask_flat, x.shape)  # reshape to original shape

        # apply the mask on the batch input
        x = x * mask

        # calculate mean (mu) and stddev (sigma square) of all non-zero elements
        # see p87
        num_non_zero = x.numel() - len(zero_indices)
        mu = ((x * mask).sum() / num_non_zero).item()
        sigma2 = (torch.sqrt(((x - mu) * mask) ** 2).sum() / num_non_zero).item()
        epsilon = 10**-5
        xhat = (x - mu) / math.sqrt(sigma2 + epsilon)

        # calc final Y
        y = gamma * xhat + beta

        # print(f"+ DropNorm.forward() result: {y.shape}")
        # print("---------------------------------")
        # NOTE: when predicting: no gamma, beta
        return y



if __name__ == '__main__':
    print("ex 2")

    batch_size = 10
    features_num = 4 * 2  # ensure even number (that can split more than once)
    N = batch_size
    M = features_num

    dn1 = DropNorm()
    t1 = torch.ones(size=(batch_size, 1, features_num))
    print(f"t1: {t1.shape}")
    ret = dn1(t1)
    print(f"ret: {ret.shape}")


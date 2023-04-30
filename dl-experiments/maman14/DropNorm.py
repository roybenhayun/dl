import math

import torch
from torch import nn
import numpy as np


class DropNorm(nn.Module):
    def __init__(self, bn_size, p=0.5):
        super().__init__()
        print(f"* init DropNorm p:{p}")
        self.p = p
        # declare as network parameters so will be learnt (p87)
        self.gamma = nn.Parameter(torch.ones(bn_size))
        self.beta = nn.Parameter(torch.zeros(bn_size))
        self.running_mu = 0
        self.running_sigma2 = 0

    def forward(self, x):
        # print("---------------------------------")
        # print(f"+ DropNorm.forward({batch_input.shape})")

        if self.training:  # see p112
            # create binary mask with half random elements as zeros
            rand_indices = torch.randn(x.shape)
            mask_flat = torch.reshape(torch.ones(size=rand_indices.shape), (-1,))  # flatten
            # generate a random sample from indices [0 .. mask_flat.numel()]
            zero_indices = np.random.choice(mask_flat.numel(), int(rand_indices.numel() * self.p), replace=False)
            mask_flat[zero_indices] = 0  # create the 1\0 mask
            mask = torch.reshape(mask_flat, x.shape)  # reshape to original shape

            # apply the mask on the batch input
            x = x * mask
        else:
            # see p113: weaken the signal
            x = (1 - self.p) * x

        # calculate mean (mu) and stddev (sigma square) of all non-zero elements
        # see p87
        if self.training:
            num_non_zero = x.numel() - len(zero_indices)
            mu = ((x * mask).sum() / num_non_zero)
            sigma2 = (torch.sqrt(((x - mu) * mask) ** 2).sum() / num_non_zero)
            self.running_mu = 0.9 * self.running_mu + 0.1*mu
            self.running_mu = 0.9 * self.running_sigma2 + 0.1 * sigma2
            epsilon = 10**-5
        else:
            # see p88: alternative mu,sigma2 for eval mode
            mu = self.running_mu
            sigma2 = self.running_sigma2
        xhat = (x - mu) / math.sqrt(sigma2 + epsilon)

        # calc final Y
        y = self.gamma * xhat + self.beta

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


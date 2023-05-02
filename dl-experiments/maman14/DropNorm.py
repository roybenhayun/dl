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
        epsilon = 10 ** -5
        if self.training:
            # compute the mean and variance along the first dimension using the dim parameter
            # use keepdim parameter to ensure tensors have a size of [batch#, 1]

            # compute the mean of non-zero elements
            # mu = torch.sum(x, dim=1, keepdim=True) / torch.sum(x != 0, dim=1, keepdim=True)
            # compute the variance of non-zero elements
            # sigma2 = torch.sum((x - mu) ** 2, dim=1, keepdim=True) / torch.sum(x != 0, dim=1, keepdim=True)

            mu = x.mean(dim=1, keepdim=True)
            sigma2 = x.var(dim=1, keepdim=True)
            self.running_mu = 0.9 * self.running_mu + 0.1*mu
            self.running_sigma2 = 0.9 * self.running_sigma2 + 0.1 * sigma2
        else:
            # see p88: alternative mu,sigma2 for eval mode
            mu = self.running_mu
            sigma2 = self.running_sigma2
        #xhat = (x - mu) / torch.sqrt(sigma2 + epsilon)
        xhat = torch.where(x!=0, (x - mu) / torch.sqrt(sigma2 + epsilon), x)

        # calc final Y
        #y = self.gamma * xhat + self.beta
        y = torch.where(x!=0, self.gamma * xhat + self.beta, xhat)
        return y


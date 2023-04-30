import math

import torch
from torch import nn
import numpy as np


class DropNorm(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        print(f"* init DropNorm p:{p}")
        self._p = p

    def forward(self, batch_input):
        print("---------------------------------")
        print(f"+ DropNorm.forward({batch_input.shape})")
        y = batch_input.clone()

        # create binary mask with half random elements as zeros
        x = torch.randn(batch_input.shape)
        mask_flat = torch.reshape(torch.ones(size=x.shape), (-1,))
        zero_indices = np.random.choice(mask_flat.numel(), int(x.numel() * self._p), replace=False)
        mask_flat[zero_indices] = 0
        mask = mask_flat.view(batch_input.shape)
        # apply mask
        y = y * mask

        # calculate mean and stddev of all non-zero elements (can't use torch.mean \ std)
        num_non_zero = y.numel() - len(zero_indices)
        mean = ((y * mask).sum() / num_non_zero).item()
        stddev = (torch.sqrt(((y - mean) * mask) ** 2).sum() / num_non_zero).item()
        epsilon = 0.0001
        y = (y - mean) / math.sqrt(stddev + epsilon)

        # calc final Y
        yi = 1
        bi = 0
        y = yi * y + bi

        print(f"+ DropNorm.forward() result: {y.shape}")
        print("---------------------------------")
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


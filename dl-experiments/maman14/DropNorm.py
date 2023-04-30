import torch
from torch import nn
import numpy as np


class DropNorm(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        print(f"* init DropNorm p:{p}")

    def forward(self, batch_input):
        print("---------------------------------")
        print(f"+ DropNorm.forward({batch_input.shape})")
        y = batch_input.clone()

        # zero half of the elements
        # Create a tensor of any size with random values
        tensor_size = (2, 3, 4)  # Example size
        x = torch.randn(batch_input.shape)

        # create binary mask with random zeros
        mask_flat = torch.reshape(torch.ones(size=x.shape), (-1,))
        zero_indices = np.random.choice(mask_flat.numel(), x.numel() // 2, replace=False)
        mask_flat[zero_indices] = 0
        mask = mask_flat.view(batch_input.shape)

        # apply mask
        y = y * mask

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


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
        # כלומר לפני שאתה מכניס לשכבה (האיפוס, הנרמול וכו) אתה מכפיל את כל הMים בכל שורה ויוצר קלט דו מימדי?
        # כן זה מה שעשיתי. מצאתי את מספר הפיצ'רים והמרתי כל sample לווקטור

        features_wo_batches = x[0]
        if self.training:  # see p112
            # create binary mask with half random elements as zeros
            # rand_indices = torch.ones(features_wo_batches)
            mask_flat = torch.reshape(torch.ones(size=features_wo_batches.shape), (-1,))  # flatten
            # generate a random sample from indices [0 .. mask_flat.numel()]
            zero_indices = np.random.choice(mask_flat.numel(), int(features_wo_batches.numel() * self.p), replace=False)
            mask_flat[zero_indices] = 0  # create the 1\0 mask
            mask = torch.reshape(mask_flat, features_wo_batches.shape)  # reshape to original shape

            # apply the mask on the batch input (use broadcasting)
            x = x * mask
        else:
            # see p113: weaken the signal
            #x = (1 - self.p) * x
            pass

        # calculate mean (mu) and stddev (sigma square) of all non-zero elements
        # see p87
        epsilon = 10 ** -5
        if self.training:
            x2 = x[:, mask.bool()] # <- will be without the zeroed features , or mask_flat...
            mu = x2.mean(dim=1, keepdim=True)
            sigma2 = x2.var(dim=1, keepdim=True)
            # see https://github.com/Idan-Alter/OU-22961-Deep-Learning/blob/main/22961_3_2_1_normalization.ipynb
            with torch.no_grad():
                self.running_mu = 0.9 * self.running_mu + 0.1*mu.mean()
                self.running_sigma2 = 0.9 * self.running_sigma2 + 0.1 * sigma2.mean()
            x = x / (1 - self.p)

        else:
            # see p88: alternative mu,sigma2 for eval mode
            mu = self.running_mu
            sigma2 = self.running_sigma2
        xhat = (x - mu) / torch.sqrt(sigma2 + epsilon)

        # calc final Y
        y = self.gamma * xhat + self.beta
        return y


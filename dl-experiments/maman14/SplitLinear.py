import torch
from torch import nn
import math


class SplitLinear(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.N = N
        self.M = M

        self.linear1 = nn.Linear(int(self.M/2), int(self.M/2))

        #init parameters. see p93 - uniform distribution in [sqrt(-1/Kin), sqrt(1/Kin)] where Kin == M/2
        start = -(1/math.sqrt(int(self.M/2)))
        end = (1 / math.sqrt(int(self.M / 2)))
        torch.nn.init.uniform_(self.linear1.weight, start, end)

        self.relu1 = nn.ReLU()

        print("---------------------------------")
        print("* init module")
        print("SplitLinear:")
        print(f"N (batch size): {self.N}, M (features num): {self.M}")
        print(f"Z1 -> Y1")
        print(f"Z1[{self.linear1}] -> Y1[{self.relu1}]")
        print(f"Z1 Linear layer weights: {self.linear1.weight.shape}, parameters uniform dist: [{start}, {end}] (see p93)")
        print("---------------------------------")

    def forward(self, batch_input):
        print("---------------------------------")
        print("* forward pass (train N.A.)")
        print("IMPL-NOTE: no training needed => no optimizer e.g., SGD, loss functions e.g., CE ")
        split = torch.split(batch_input, int(self.M / 2), dim=1)
        print(f"split input to 2 chunks: {split[0].shape}, {split[1].shape}")

        r1 = self.linear1(split[0])
        print(f"Z1 output for chunk 1: {r1.shape}")
        r2 = self.linear1(split[1])
        print(f"Z1 output for chunk 2: {r2.shape}")
        r3 = self.relu1(r1)
        print(f"Y1 output for chunk 1: {r3.shape}")
        r4 = self.relu1(r2)
        print(f"Y1 output for chunk 2: {r4.shape}")
        Y = torch.cat((r3, r4), dim=1)
        print(f"concatenate Y1 outputs: {Y.shape}")

        print(f"forward pass result: {Y.shape}")
        print("---------------------------------")
        return Y


if __name__ == '__main__':
    print("ex 1")

    batch_size = 10
    features_num = 4 * 2  # ensure even number (that can split more than once)
    N = batch_size
    M = features_num

    s1 = SplitLinear(batch_size, features_num)
    t1 = torch.ones(size=(batch_size, features_num))
    ret = s1(t1)


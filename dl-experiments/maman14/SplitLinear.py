import torch
from torch import nn
import math
from tqdm import tqdm
import sklearn.datasets as skds
import matplotlib.pyplot as plt

class SplitLinear(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.N = N
        self.M = M

        self.linear1 = nn.Linear(int(self.M/2), int(self.M/2))
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
        if self.training:
            print("---------------------------------")
            print("* forward pass (train N.A.)")
            Y = torch.zeros(size=(N, M))
            split = torch.split(batch_input, int(self.M/2), dim=1)
            print(f"split input to 2 chunks: {split[0].shape}, {split[1].shape}")
            for idx in range(N):
                print(f"pass Z1 for batch {idx}")
                r1 = self.linear1(split[0][idx])
                print(f"Z1 output for chunk 1: {r1.shape}")
                r2 = self.linear1(split[1][idx])
                print(f"Z1 output for chunk 2: {r2.shape}")
                print(f"pass Y1 for batch {idx}")
                r3 = self.relu1(r1)
                print(f"Y1 output for chunk 1: {r3.shape}")
                r4 = self.relu1(r2)
                print(f"Y1 output for chunk 2: {r4.shape}")
                Y[idx] = torch.cat((r3, r4))
                print(f"concatenate Y1 outputs: {Y[idx].shape}")

            print(f"forward pass result: {Y.shape}")
            print("---------------------------------")
            return Y
        else:
            print("eval..")

    def reset_parameters_impl(self):
        pass


if __name__ == '__main__':
    print("ex 1")

    batch_size = 10
    features_num = 3 * 2  # ensure even number
    N = batch_size
    M = features_num

    s1 = SplitLinear(batch_size, features_num)
    t1 = torch.ones(size=(batch_size, features_num))
    ret = s1(t1)


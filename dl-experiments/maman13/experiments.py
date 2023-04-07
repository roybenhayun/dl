import torch
import sklearn.datasets as skds
import matplotlib.pyplot as plt

X, Y = skds.make_blobs(n_samples=100, n_features=2,
                       centers=2, random_state=1)
X = torch.tensor(X).float()
Y = torch.tensor(Y).float()

from torch import nn
z = nn.Linear(2,1)
y = nn.Sigmoid()

print(f"z.weight: {z.weight} \nz.bias: {z.bias}")


print(f"X[0,:] = {X[0,:]}")
print(f"z(X[0,:]) = {z(X[0, :])}")
print(f"y(z(X[0,:])) = {y(z(X[0,:]))}")
a = z.weight[0,0]*X[0,0]+z.weight[0,1]*X[0,1]+z.bias[0]
b = z(X[0,:])
print(f"a: {a} \nb: {b}")
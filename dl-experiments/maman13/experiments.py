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

# https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
m = nn.Softmax(dim=1)  # dimension along which *Softmax will be computed*
input = torch.randn(2, 3)
print(input, input.size())
output = m(input)  # returns Tensor with same size,  values in range [0, 1]
print(output, output.size())

# https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
m = nn.Linear(20, 30) # input 20, output 30
input = torch.randn(128, 20) # 128 rows of 20 cols
output = m(input)
print(output.size())

s = nn.Softmax(dim=1)
activation = s(output)
print(activation.size())
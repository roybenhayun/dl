import torch
import torch.nn as nn

def print_tensor(name, t):
    print("_______________________")
    print(f"{name} numel: {t.numel()}, dtype: {t.dtype}")
    print(f"{name} shape: {t.shape}")
    print(f"{name} data: {t}")
    print("_______________________")

t1 = torch.ones((2,2,2))
print_tensor("t1", t1)

b = torch.arange(1 * 2 * 3).view(1, 2, 3)
print_tensor("b", b)
sb = torch.sum(b, (2, 1))
print_tensor("sb", sb)


# target output size of 5x7
m = nn.AdaptiveAvgPool2d((5, 7))
input = torch.randn(1, 64, 8, 9)
output = m(input)
print_tensor("output", output)

# target output size of 7x7 (square)
m = nn.AdaptiveAvgPool2d(7)
input = torch.randn(1, 64, 10, 9)
output = m(input)
print_tensor("output", output)

# target output size of 10x7
m = nn.AdaptiveAvgPool2d((None, 7))
input = torch.randn(1, 64, 10, 9)
output = m(input)
print_tensor("output", output)
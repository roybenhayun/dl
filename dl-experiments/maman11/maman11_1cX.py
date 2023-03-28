import torch
from utils import print_tensor
from maman11_1b import get_broadcastable

A = torch.arange(3).reshape(3,1)
B = torch.arange(4).reshape(1,4)
print_tensor("A", A)
print_tensor("B", B)
print_tensor("A+B", A+B)
b, s = get_broadcastable(A, B)
print(f"A2B: b: {b}, size: {s}")
b, s = get_broadcastable(B, A)
print(f"B2A: b: {b}, size: {s}")

try:
    A.expand_as(B)
except Exception as e:
    print(f"A.expand_as(B) failed: {e}")

try:
    B.expand_as(A)
except Exception as e:
    print(f"B.expand_as(A) failed: {e}")

print("********************************")
#x = torch.arange(1).view(1, 1)
x = torch.arange(3).view(1, 3)
y = torch.arange(2).view(2, 1)
print_tensor("x", x)
print_tensor("y", y)

try:
    e1 = x.expand_as(y)  # singleton dimensions expanded to a larger size.
    print(f"x.expand_as(y) works: {e1.shape}")
except Exception as e:
    print(f"x.expand_as(y) failed: {e}")

try:
    e2 = y.expand_as(x)
    print(f"y.expand_as(x) works: {e2.shape}")
except Exception as e:
    print(f"y.expand_as(x) failed: {e}")

a, b = torch.broadcast_tensors(x, y)  # [1, 3], [2, 1]
print_tensor("a", a)  # [2, 3]
print_tensor("b", b)  # [2, 3]

print("********************************")

x = torch.arange(3).view(1, 3)
y = torch.arange(2).view(2, 1)

try:
    e1 = x.expand(5, 1, 4)
    print(f"x.expand_as(y) works: {e1.shape}")
except Exception as e:
    print(f"x.expand_as(y) failed: {e}")

try:
    e2 = y.expand_as(x)
    print(f"y.expand_as(x) works: {e2.shape}")
except Exception as e:
    print(f"y.expand_as(x) failed: {e}")
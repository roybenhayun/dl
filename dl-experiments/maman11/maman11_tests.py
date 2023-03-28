import torch
from maman11_1a import get_broadcastable

x=torch.empty(5,7,3)
y=torch.empty(5,7,3)
b, s = get_broadcastable(x, y)
print(f"x2y broadcastable: b: {b}, size: {s}")

x=torch.empty((0,))
y=torch.empty(2,2)
b, s = get_broadcastable(x, y)
print(f"x2y not broadcastable : b: {b}, size: {s}")

x=torch.empty(5,3,4,1)
y=torch.empty(  3,1,1)
b, s = get_broadcastable(x, y)
print(f"x2y broadcastable: b: {b}, size: {s}")  # TODO: FIX

x=torch.empty(5,2,4,1)
y=torch.empty(  3,1,1)
b, s = get_broadcastable(x, y)
print(f"x2y not broadcastable: b: {b}, size: {s}")

B = torch.ones(size=(1, 2, 1))
A = torch.ones(size=(2, 3, ))
b, s = get_broadcastable(A, B)
print(f"A2B broadcastable: b:{b}, size: {s}")

A = torch.ones(size=(1, 2, ))
B = torch.ones(size=(3, 1, 2))
b, s = get_broadcastable(A, B)
print(f"A2B broadcastable: b:{b}, size: {s}")

A = torch.arange(0, 4).reshape(2, 2)
B = torch.ones(size=(3, 2, 2))
b, s = get_broadcastable(A, B)
print(f"A2B broadcastable: b:{b}, size: {s}")
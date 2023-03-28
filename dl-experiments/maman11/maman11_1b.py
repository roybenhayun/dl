import torch
from utils import print_tensor
from maman11_1a import get_broadcastable

x = torch.randn(1, 3)
torch.cat((x, x, x), 0)

A = torch.arange(0, 2).reshape(1, 2)
B = torch.ones(size=(3, 2, 2))
print_tensor("A", A)
Ae = A.expand_as(B)
print_tensor("Ae", Ae)
b, s = get_broadcastable(A, B)
print(f"********** broadcast: {b}, to: {s}")
dim_dif = len(s) - len(A.shape)
print(f"dim diff = {dim_dif}")
Asq = A
while dim_dif > 0:
    dim_dif -= 1
    Asq = torch.unsqueeze(Asq, 0)
    dim_size = s[dim_dif]
    l = [Asq[0]] * dim_size
    Asq = torch.stack(l, 0)


print_tensor("Asq", Asq)
print(f"equal: {torch.equal(Ae, Asq)}")

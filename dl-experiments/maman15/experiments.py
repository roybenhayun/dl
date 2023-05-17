import torch

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
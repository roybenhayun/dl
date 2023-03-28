import torch

def print_tensor(name, t):
    print("_______________________")
    print(f"{name} numel: {t.numel()}, dtype: {t.dtype}")
    print(f"{name} shape: {t.shape}")
    print(f"{name} data: {t}")
    print("_______________________")
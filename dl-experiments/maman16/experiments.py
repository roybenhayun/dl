import torch

def my_hook(x):
    return torch.ones(x.size())

v = torch.tensor([0., 0., 0.], requires_grad=True)
print(v)
h = v.register_hook(my_hook)  # double the gradient
v.backward(torch.tensor([1., 2., 3.]))
print(v.grad)


h.remove()  # removes the hook
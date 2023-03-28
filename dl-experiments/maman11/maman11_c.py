import torch
from utils import print_tensor
from maman11_1a import get_broadcastable, expand_as_impl

def broadcast_tensors_impl(A, B):
    '''
    If two tensors x, y are broadcastable:
        prepend 1 to the dimensions of the shorter tensor
        dimension size is the max of the sizes of x and y along that dimension.

    :param A:
    :param B:
    :return: A broadcasted to B, B broadcasted to A
    '''
    b1, s1 = get_broadcastable(A, B)

    if b1:
        Ae = expand_as_impl(A, torch.empty(size=s1))

    b2, s2 = get_broadcastable(B, A)
    if b2:
        Be = expand_as_impl(B, torch.empty(size=s2))
    return Ae, Be

x = torch.arange(3).view(1, 3)
y = torch.arange(2).view(2, 1)
a1, b1 = torch.broadcast_tensors(x, y)
print_tensor("a1", a1)
print_tensor("b1", b1)


a2, b2 = broadcast_tensors_impl(x, y)
print_tensor("a2", a2)
print_tensor("b2", b2)
print(f"equal: {torch.equal(a1, a2)}")
print(f"equal: {torch.equal(b1, b2)}")

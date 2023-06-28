import torch
import numpy
#
# B = torch.ones(size=(1, 1))
# A = torch.ones(size=(1, 1, 1, ))
# a_dims = len(A.shape) #1
# b_dims = len(B.shape) #2
# if a_dims > b_dims:
#     print("A dimensions bigger then B dimensions")
# A.expand_as(B)


def expand_as_impl(A, B):
    '''
    See https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html

    singleton dimensions expanded to a larger size.
    new dimensions will be appended at the front

    :param A: tensor
    :param B: tensor
    :return: A expanded to dimensions of B
    '''

    s_a = A.shape
    s_b = B.shape

    #check sizes
    if len(s_a) > len(s_b):
        raise Exception(f"the number of sizes provided ({len(s_a)}) must be greater or equal to the number of dimensions in the tensor ({len(s_b)})")

    # verify match the existing size at non-singleton dimension
    i = 1
    while i <= len(s_a):
        last_a = s_a[len(s_a) - i]
        last_b = s_b[len(s_b) - i]
        if last_a != 1 and last_a != last_b:
            raise Exception(f"The expanded size of the tensor ({last_b}) must match the existing size ({last_a}) at non-singleton dimension {i-1}.  Target sizes: [{last_b}].  Tensor sizes: [{last_a}]")
        i += 1

    # append new dimensions
    dim_dif = len(s_b) - len(s_a)
    Ae = A
    while dim_dif > 0:
        dim_dif -= 1
        Ae = torch.unsqueeze(Ae, 0)
        dim_size = s_b[dim_dif]
        l = [Ae[0]] * dim_size
        Ae = torch.stack(l, 0)

    # enlarge singleton dimensions
    for i, dt in enumerate(list(s_b)):
        da = Ae.shape[i]
        if(dt > da):
            l = [Ae] * dt  # da is 1. list trick, same as ['a', 'b'] * 3 = ['a', 'b', 'a', 'b', 'a', 'b']
            Ae = torch.cat(l, dim=i)

    return Ae


def __is_broadcastable(A, B) -> bool:

    # 1. Each tensor has at least one dimension
    if len(A.size()) == 0 or len(A.size()) == 0:
        return False

    # 2. Iterate
    # sizes must either be equal, one of them is 1, or one of them does not exist
    a_dims = list(A.size())
    b_dims = list(B.size())
    # setup to iterate on shorter tensor
    shorter = a_dims
    longer = b_dims
    if len(a_dims) > len(b_dims):
        tmp = b_dims
        longer = a_dims
        shorter = tmp
    l_indx = len(longer) - 1
    s_indx = len(shorter) - 1
    while l_indx >= 0 and s_indx >= 0:
        if not (longer[l_indx] == shorter[s_indx] or longer[l_indx] == 1 or shorter[s_indx] == 1):
            return False
        l_indx -= 1
        s_indx -=1
    return True


def __calc_broadcasting_size(A, B):
    a_dims = list(A.size())
    b_dims = list(B.size())

    # add 1's to beginning of shorter tensor
    while len(a_dims) < len(b_dims):
        a_dims[:0] = [1]
    while len(b_dims) < len(a_dims):
        b_dims[:0] = [1]

    i = len(a_dims) - 1
    match = True
    dimensions = []
    # add dimension to result and adjust singleton dimensions
    while match and i >= 0:
        if a_dims[i] == b_dims[i]:
            dimensions.append(b_dims[i])
        elif a_dims[i] == 1:
            dimensions.append(b_dims[i])
        elif b_dims[i] == 1:
            dimensions.append(a_dims[i])
        else:
            raise Exception("should not happen at this point")
        i -= 1

    dimensions.reverse()
    return torch.Size(dimensions)


def get_broadcastable(A, B) -> tuple:
    '''
    See https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
    :param A: tensor
    :param B: tensor
    :return: tuple with true\false if A is can be broadcast to B, and eventual dimensions if true
    '''

    broadcastable = __is_broadcastable(A, B)
    if not broadcastable:
        return False
    else:
        return True, __calc_broadcasting_size(A, B)


x = torch.tensor([[1], [2], [3]])
y = torch.empty(3, 4)
ret = expand_as_impl(x, y)
print(ret.shape)

x = torch.tensor([3, 3])  # vs x = torch.empty(3, 3) === x2 = torch.tensor([[3,3,3],[3,3,3],[3,3,3]])
y = torch.tensor([2])  # vs y = torch.empty(2)
passed_1st_stage = False
try:
    y.expand_as(x)  # should pass
    passed_1st_stage = True
    x.expand_as(y)  # should fail
except Exception as e:
    assert passed_1st_stage, "expected to pass first variant"
    print("threw error as expected: ", e)

passed_1st_stage = False
try:
    ret = expand_as_impl(y, x)  # should pass
    passed_1st_stage = True
    ret = expand_as_impl(x, y)  # should fail
    print("error: ", ret.shape)
except Exception as e:
    assert passed_1st_stage, "expected to pass first variant"
    print("threw error as expected: ", e)




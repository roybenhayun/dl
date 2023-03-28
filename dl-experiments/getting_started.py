import torch
from utils import print_tensor

#
# https://d2l.ai/chapter_preliminaries/ndarray.html
#

def getting_started_1():
    """
    https://d2l.ai/chapter_preliminaries/ndarray.html#getting-started

    :return:
    """
    x = torch.arange(12, dtype=torch.float32)
    x
    x.numel()
    x.shape
    #change the shape of a tensor
    X = x.reshape(3, 4)
    X = x.reshape(3, -1)

    torch.zeros((2, 3, 4))
    torch.ones((2, 3, 4))
    torch.randn(3, 4)
    torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

    print("---------------------------------")
    print("Indexing_and_Slicing")
    X[-1]
    X[1:3]
    X[1, 2] = 17
    X
    #first and second rows \ elements along axis 1
    X[:2, :] = 12
    X

    print("---------------------------------")
    print("2.1.3. Operations")
    print(torch.exp(x))
    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    x = torch.tensor([1.0, 2, 4, 8])
    y = torch.tensor([2, 2, 2, 2])
    x + y, x - y, x * y, x / y, x ** y

    X = torch.arange(12, dtype=torch.float32).reshape((3,4))
    Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    torch.cat((X, Y), dim=0)
    torch.cat((X, Y), dim=1)

    print(X == Y)
    print(X.sum())

    print("---------------------------------")
    print("2.1.4. Broadcasting")
    broadcasting()

    print("---------------------------------")
    print("2.1.5. Saving Memory")
    before = id(Y)
    print(before)
    Y = Y + X
    print(id(Y), id(Y) == before)

    #updates in place - use X[:] = X + Y or X += Y
    Z = torch.zeros_like(Y)
    print('id(Z):', id(Z))
    Z[:] = X + Y  # using slice notation
    print('id(Z):', id(Z))

    before = id(X)
    X += Y  # using +=
    print(id(X) == before)

    print("---------------------------------")
    print("2.1.6. Conversion to Other Python Objects")
    A = X.numpy()
    B = torch.from_numpy(A)
    print(type(A))
    print(type(B))

    a = torch.tensor([3.5])
    print_tensor("a", a)
    print(f"{a.item()}, {float(a)}, {int(a)}")


def broadcasting():
    #https://pytorch.org/docs/stable/notes/broadcasting.html
    print_tensor("arange", torch.arange(3))
    print_tensor("arange", torch.arange(2))
    a = torch.arange(3).reshape((3, 1))
    b = torch.arange(2).reshape((1, 2))
    print_tensor("a", a)
    print_tensor("b", b)
    c = a + b
    print_tensor("a+b", c)




def exercises():
    print("------------------------------------------")
    print("2.1.8. Exercises")
    print("------------------------------------------")
    x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print_tensor("x", x)
    print_tensor("y", y)
    print_tensor("x == y", x == y)
    print_tensor("x > y", x > y)
    print_tensor("x < y", x < y)

    az = 3
    bz = 3
    a = torch.arange(3*az).reshape((3, 1, az))
    b = torch.arange(2*bz).reshape((1, 2, bz))
    print_tensor("a", a)
    print_tensor("b", b)
    c = a * b
    print_tensor("a+b", c)



def unit_1():
    x = torch.tensor([[1.0, 2], [3, 4]])
    print(x, x.shape, x.dtype, sep='\n')

    x = torch.arange(9)
    y = torch.eye(3, dtype=torch.bool)
    z = torch.randn(size=[2, 3, 3])
    print(x, y, z, sep='\n')

    list = [1, 1.0, 0]
    x = torch.tensor(list)
    # list = [1, 1.0, 'o']
    # x = torch.tensor(list)  # TypeError: new(): invalid data type 'str'

    ex2 = torch.randint(low=0, high=2, size=(2, 2, 2, 2))
    print_tensor("ex2", ex2)




if __name__ == '__main__':
    # getting_started_1()
    # exercises()
    unit_1()

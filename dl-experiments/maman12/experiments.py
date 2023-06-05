import torch

from utils import print_tensor

#assert, torch.rand, torch.cumsum, tensor.sum(), tensor.all()

number = 1
assert number > 0, f"number greater than 0 expected, got: {number}"

#
# torch.rand
# Returns a tensor filled with random numbers from a uniform distribution on the interval [0,1)
#

tr1 = torch.rand(5)
print_tensor("tr1", tr1)

tr2 = torch.rand(2, 3)
print_tensor("tr2", tr2)


#
# torch.cumsum
# Returns the cumulative sum of elements of input in the dimension dim.
# e.g., yi = x1 + x2 + x3 + ⋯ + xi
#

a = torch.randn(10)
print_tensor("a", a)
as1 = torch.cumsum(a, dim = 0)
print_tensor("as1", as1)

a2 = torch.arange(10)
print_tensor("a2", a2)
as2 = torch.cumsum(a2, dim=0)
print_tensor("as2", as2)
as3 = torch.cumsum(torch.reshape(a2, (5, 2)), dim=1)
print_tensor("as3", as3)

#
# torch.sum
# Returns the sum of all elements in the input tensor.
#

as4 = torch.sum(a2)
print_tensor("as4", as4)


a3 = torch.reshape(torch.arange(9), (3, 3))
print_tensor("a3", a3)
as5 = torch.sum(a3)
print_tensor("as5", as5)


#
# torch.sum(t, dim)
#       dim (int or tuple of ints, optional) – the dimension or dimensions to reduce
#

x = torch.tensor([[ 0.9569, -0.6598],
                  [ 0.9742, -1.0970],
                  [-0.3451, 0.7615],
                  [-0.8656, 1.8823],
                  [ 0.6175, 0.2272],
                  [ 0.9894, -0.8952],
                  [ 1.0751, -1.2522]])

x = torch.tensor([[ 1, 2],
                  [ 1, 2],
                  [ 1, 2],
                  [ 1, 2],
                  [ 1, 2],
                  [ 1, 2],
                  [ 1, 2]])
print(x.shape, x)
s1 = torch.sum(x)
print(s1.shape, s1)
s2 = torch.sum(x, dim=0)
print(s2.shape, s2)
s3 = torch.sum(x, dim=1)
print(s3.shape, s3)

b = torch.arange(2 * 3 * 4).view(2, 3, 4)
print(b.shape, b)
sb = torch.sum(b, (1,2))
print(sb.shape, sb)
#
# torch.all
# Tests if all elements in input evaluate to True.
#

a = torch.reshape(torch.arange(0, 16), (4, 4))
print_tensor("a", a)

aa1 = torch.all(a)
print_tensor("aa1", aa1)

#
# Size
#

s1 = torch.Size()
out = torch.ones(s1)
print(s1, out)
s1 = torch.Size(())
out = torch.ones(s1)
print(s1, out)
#s1 = torch.Size(1)  # TypeError: 'int' object is not iterable
#s1 = torch.Size((1))  # TypeError: 'int' object is not iterable
s1 = torch.Size((0, ))
out = torch.ones(s1)
print(s1, out)
s1 = torch.Size((0, 1))
out = torch.ones(s1)
print(s1, out)
s1 = torch.Size((1, ))
out = torch.ones(s1)
print(s1, out)
s1 = torch.Size((1, 1))
out = torch.ones(s1)
print(s1, out)

s1 = 10
s2 = (2, 8)



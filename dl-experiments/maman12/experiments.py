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



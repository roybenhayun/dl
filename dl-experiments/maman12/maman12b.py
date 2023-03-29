import torch

#
# b1
#


class MyScalar:

    def __init__(self, value, grad=None, prev=None):
        self.value = value
        self.grad = grad
        self.prev = prev


#
# b2
#

def exp(a) -> MyScalar:
    t = torch.exp(torch.Tensor([a.value]))
    out = MyScalar(t.item(), t.item(), a)
    return out


def ln_a(scalar) -> MyScalar:
    pass


def sin_a(scalar) -> MyScalar:
    pass


def cos_a(scalar) -> MyScalar:
    pass


def power(a, b) -> MyScalar:
    t = torch.pow(torch.Tensor([a.value]), torch.Tensor([b]))
    out = MyScalar(t.item(), b * (a.value ** (b-1)), a)
    return out



def n_times_a(scalar) -> MyScalar:
    pass


def a_plus_n(scalar) -> MyScalar:
    pass


#
# b3
#

dict={}

def __traverse_computational_graph(scalar, ext) -> float:
    print(f"__calc_chain_rule: scalar.value={scalar.value}, scalar.grad={scalar.grad}, ext={ext} ")
    if scalar.prev is None:
        print(f"LEAF: scalar.value={scalar.value}, scalar.grad={scalar.grad}, ext={ext}")
        dict[scalar] = ext
        return
    else:
        print(f"PRE: scalar.value {scalar.value}")
        d = ext * scalar.grad  # external * internal
        dict[scalar] = ext
        __traverse_computational_graph(scalar.prev, d)
        print(f"POST: scalar.value {scalar.value}")
        return

def get_gradient(scalar) -> dict:
    '''
    apply the Chain Rule
    '''
    __traverse_computational_graph(scalar, 1)
    print("dict ", dict)
    return dict
    # if scalar.prev is None:
    #     print("dx\dx: ", 1)
    #     dict[scalar] = 1
    #     return 1
    # else:
    #     partial = get_gradient(scalar.prev)
    #     derivative = scalar.grad * partial
    #     print(f"dy\dx: {derivative}, grad: {scalar.grad}")
    #     dict[scalar] = derivative
    #     return derivative


if __name__ == '__main__':
    print("------------------------------")
    ta = torch.tensor([2.], requires_grad=True)
    print(f"ta: {ta}")
    tb = torch.pow(ta, 2.)
    print(f"tb: {tb}")
    tb.retain_grad()
    tc = torch.exp(tb)
    print(f"tc: {tc}")
    tc.retain_grad()
    td = tc.backward()
    print("ta.grad: ", ta.grad)
    print("tb.grad: ", tb.grad)
    print("tc.grad: ", tc.grad)
    print("------------------------------")

    a = MyScalar(2)  # a = 2
    b = power(a, 2)  # a^2 db\da = 2a
    c = exp(b)  # dc\db = e^b = e^(2a)
    d = get_gradient(c)  # calc by chain rule
    # FIXME: in opposite directions from torch
    print("a.grad: ", dict[a])  # 218.3926
    print("b.grad: ", dict[b])  # 54.598150033144
    print("c.grad: ", dict[c])  # 1


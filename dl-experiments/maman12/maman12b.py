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


def ln(a) -> MyScalar:
    t = torch.log(torch.Tensor([a.value]))
    out = MyScalar(t.item(), 1 / a.value, a)
    return out


def sin(a) -> MyScalar:
    from math import cos
    t = torch.sin(torch.Tensor([a.value]))
    out = MyScalar(t.item(), cos(a.value), a)
    return out


def cos(a) -> MyScalar:
    from math import sin
    t = torch.cos(torch.Tensor([a.value]))
    out = MyScalar(t.item(), -sin(a.value), a)
    return out


def power(a, b) -> MyScalar:
    t = torch.pow(torch.Tensor([a.value]), torch.Tensor([b]))
    out = MyScalar(t.item(), b * (a.value ** (b-1)), a)
    return out



def mul(a, b) -> MyScalar:
    t = torch.mul(torch.Tensor([a.value]), torch.Tensor([b]))
    out = MyScalar(t.item(), b, a)
    return out


def add(a, b) -> MyScalar:
    t = torch.add(torch.Tensor([a.value]), torch.Tensor([b]))
    out = MyScalar(t.item(), 1, a)
    return out


#
# b3
#



def __traverse_computational_graph(scalar, ext, dict) -> float:
    print(f"__traverse_computational_graph: scalar.value={scalar.value}, scalar.grad={scalar.grad}, ext={ext} ")
    dict[scalar] = ext
    if scalar.prev is None:
        print(f"LEAF: scalar.value={scalar.value}, scalar.grad={scalar.grad}, ext={ext}")
        return
    else:
        print(f"RECURSE: scalar.value {scalar.value}")
        __traverse_computational_graph(scalar.prev, ext * scalar.grad, dict)
        return


def get_gradient(scalar) -> dict:
    dict = {}
    __traverse_computational_graph(scalar, 1, dict)
    print("dict ", dict)
    return dict


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
    print("a.grad: ", d[a])  # 218.3926
    print("b.grad: ", d[b])  # 54.598150033144
    print("c.grad: ", d[c])  # 1

    assert d[a] == ta.grad
    assert d[b] == tb.grad
    assert d[c] == tc.grad
    print("PASS")

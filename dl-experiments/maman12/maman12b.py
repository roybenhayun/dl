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

def get_gradient(scalar, dict={}) -> dict:
    '''
    apply the Chain Rule
    '''

    if scalar.prev is None:
        print("dx\dx: ", 1)
        return {scalar: 1}
    else:
        partial_dict = get_gradient(scalar.prev, {})
        derivative = scalar.grad * partial_dict[scalar.prev]
        print("dy\dx: ", derivative)
        partial_dict[scalar] = derivative
        return partial_dict


if __name__ == '__main__':
    a = MyScalar(2)  # a = 2
    b = power(a, 2)  # a^2 db\da = 2a
    c = exp(b)  # dc\db = e^b = e^(2a)
    d = get_gradient(c)  # calc by chain rule
    print(d[a])  # 1
    print(d[b])  # 54.598150033144
    print(d[c])  # 218.3926

    ta = torch.tensor([2.], requires_grad=True)
    tb = torch.pow(ta, 2.)
    tb.retain_grad()
    tc = torch.exp(tb)
    tc.retain_grad()
    td = tc.backward()
    print("ta.grad: ", ta.grad)
    print("tb.grad: ", tb.grad)
    print("tc.grad: ", tc.grad)
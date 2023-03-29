import torch
from math import fsum

import numpy as np
import matplotlib.pyplot as plt


def my_sampler(size, dist, requires_grad=False):
    # test sum of probabilities is 1
    dist_t = torch.Tensor(dist)
    # print("dist_t: ", dist_t)
    #sum = torch.sum(dist_t, dim=0)  # NOTE: PyTorch sum may cause a minor floating point error for some test cases
    sum = fsum(dist)
    assert fsum(dist) == 1, f"expected sum of probabilities 1, but is {sum}"

    # check all probabilities are positive
    all_true = torch.all(dist_t > 0)
    assert all_true == 1, f"expected only positive probabilities 1, but received {dist}"

    # cumulative probabilities
    sigma_dist_t_i = torch.cumsum(dist_t, dim=0)
    # print("cum_probs: ", sigma_dist_t_i)

    if type(size) is tuple:
        out = torch.ones(size, dtype=torch.bool) * False
    else:
        out = torch.ones((size, ), dtype=torch.bool) * False
    probs_samples = torch.rand(size=out.size())  # torch.tensor([[0.4607, 0.2123, 0.8472, 0.6048, 0.7650]])
    probs_samples_flat = torch.reshape(probs_samples, (1, out.numel()))
    # print("probs_samples: ", probs_samples)
    # print("probs_samples_flat: ", probs_samples_flat)
    p0 = dist_t[0]
    sigma_n = sigma_dist_t_i[-1]

    t_zeros = torch.where(probs_samples_flat < p0, True, False)
    t_ns = torch.where(probs_samples_flat >= sigma_n, True, False)

    # print("t_zeros: ", t_zeros)
    print("t_ns: ", torch.any(t_ns))

    out  = probs_samples_flat.clone()
    for i in range(len(dist) - 1):  # equivalent to ignoring the last sigma_dist_t_i which is '1'
        t_i_m1 = torch.where(probs_samples_flat >= sigma_dist_t_i[i], True, False)
        t_i = torch.where(probs_samples_flat < sigma_dist_t_i[i+1], True, False)
        t3 = t_i_m1 & t_i
        # print("t_i_m1: ", t_i_m1)
        # print("t_i: ", t_i)
        # print(f"t[{i}]: ", t3)
        out = torch.where(t3 == True, i+1, out)
    # print(f"out: ", out)
    out = torch.where(t_zeros == True, 0, out)
    # print(f"apply t_zeros: ", out)
    out = torch.where(t_ns == True, len(dist) - 1, out)
    # print(f"apply t_ns: ", out)

    if type(size) is tuple:
        out = torch.reshape(out, size)
    else:
        out = torch.reshape(out, (size, ))
    out.requires_grad_(requires_grad)
    return out



def render_10k_samples():
    dist = [0.1, 0.2, 0.7]
    A = my_sampler(10000, dist, requires_grad=False)
    print(A, A.grad, sep='\n')
    render_histogram(A, dist)


def render_histogram(A, dist):
    n, bins, patches = plt.hist(np.asarray(A.storage()), len(dist))
    ax = plt.gca()
    ax.set_ylim([0, A.numel()])
    plt.xticks(range(0, len(dist)))

    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')
    plt.title(f'dist: {dist}, samples: {A.numel()}', fontweight="bold")
    plt.show()


if __name__ == '__main__':
    render_10k_samples()
    print("-----------------------------")
    A = my_sampler(10, [0.5, 0.5])
    print(A, A.grad, sep='\n')
    render_histogram(A, [0.5, 0.5])
    print("-----------------------------")

    print("\n-----------------------------")

    A = my_sampler((2, 8), [0.1, 0.3, 0.6], requires_grad=False)
    print(A, A.grad, sep='\n')
    render_histogram(A, [0.1, 0.3, 0.6])
    print("-----------------------------")

    print("-----------------------------")
    ret = my_sampler(10, [0.5, 0.5], requires_grad=True)
    print("ret: ", ret)
    print("-----------------------------")

    print("\n-----------------------------")
    A = my_sampler((2, 2, 2), [0.3, 0.3, 0.4], requires_grad=True)
    print(A.size(), A.grad, A, sep='\n')
    print("-----------------------------")

    try:
        my_sampler(10, [0.5, 0.1])
        print("FAIL")
    except AssertionError as ae:
        print(f"PASS\ncaught as expected: {ae}")

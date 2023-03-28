import torch

from utils import print_tensor


def my_sampler(size, dist, requires_grad=False):
    # test sum of probabilities is 1
    dist_t = torch.Tensor(dist)
    print("dist_t: ", dist_t)
    sum = torch.sum(dist_t, dim=0)
    assert sum == 1, f"expected sum of probabilities 1, but is {sum}"

    # check all probabilities are positive
    all_true = torch.all(dist_t > 0)
    assert all_true == 1, f"expected only positive probabilities 1, but received {dist}"

    # cumulative probabilities
    sigma_dist_t_i = torch.cumsum(dist_t, dim=0)
    print("cum_probs: ", sigma_dist_t_i)

    if type(size) is tuple:
        out = torch.ones(size, dtype=torch.bool) * False
    else:
        out = torch.ones((size, ), dtype=torch.bool) * False
    probs_samples = torch.rand(size=out.size())  # torch.tensor([[0.4607, 0.2123, 0.8472, 0.6048, 0.7650]])
    probs_samples_flat = torch.reshape(probs_samples, (1, out.numel()))
    print("probs_samples: ", probs_samples)
    print("probs_samples_flat: ", probs_samples_flat)
    #torch.where(cum_probs >  , dis
    p0 = dist_t[0]
    sigma_n = sigma_dist_t_i[sigma_dist_t_i.numel()-1]

    t_zeros = torch.where(probs_samples_flat < p0, True, False)
    t_ns = torch.where(probs_samples_flat >= sigma_n, True, False)
    print("t_zeros: ", t_zeros)
    print("t_ns: ", t_ns)

    out  = probs_samples_flat.clone()
    for i in range(len(dist) - 1):  # equivalent to ignoring the last sigma_dist_t_i which is '1'
        t_i_m1 = torch.where(probs_samples_flat >= sigma_dist_t_i[i], True, False)
        t_i = torch.where(probs_samples_flat < sigma_dist_t_i[i+1], True, False)
        t3 = t_i_m1 & t_i
        print("t_i_m1: ", t_i_m1)
        print("t_i: ", t_i)
        print(f"t[{i}]: ", t3)
        out = torch.where(t3 == True, i+1, out)
    print(f"out: ", out)
    out = torch.where(t_zeros == True, 0, out)
    print(f"apply t_zeros: ", out)
    out = torch.where(t_ns == True, len(dist) - 1, out)
    print(f"apply t_ns: ", out)

    if type(size) is tuple:
        out = torch.reshape(out, size)
    else:
        out = torch.reshape(out, (size, ))
    out.requires_grad_(requires_grad)
    return out




if __name__ == '__main__':
    print("-----------------------------")
    ret = my_sampler(10, [0.5, 0.5])
    print("ret: ", ret)
    print("-----------------------------")

    print("-----------------------------")
    ret = my_sampler(10, [0.5, 0.5], requires_grad=True)
    print("ret: ", ret)
    print("-----------------------------")

    print("\n-----------------------------")
    A = my_sampler((2, 8), [0.3, 0.3, 0.4], requires_grad=False)
    print(A, A.grad, sep='\n')
    print("-----------------------------")

    print("\n-----------------------------")
    A = my_sampler((2, 2, 2), [0.3, 0.3, 0.4], requires_grad=True)
    print(A.size(), A.grad, A, sep='\n')
    print("-----------------------------")

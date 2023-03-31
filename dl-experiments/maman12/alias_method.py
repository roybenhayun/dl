#see https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

import numpy        as np
import numpy.random as npr

def alias_setup(probs):
    K       = len(probs)
    q       = np.zeros(K)
    J       = np.zeros(K, dtype=np.int32)

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger  = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] - (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    K  = len(J)

    # Draw from the overall uniform mixture.
    kk = int(np.floor(npr.rand()*K))

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    dice2 = npr.rand()
    if dice2 < q[kk]:
        return kk
    else:
        return J[kk]

K = 3
N = 10

# Get a random probability vector.
probs = npr.dirichlet(np.ones(K), 1).ravel()
print("probs: ", probs)

# Construct the table.
J, q = alias_setup(probs)
print("J: ", J)
print("k: ", q)

# Generate variates.
X = np.zeros(N)
for nn in range(N):
    X[nn] = alias_draw(J, q)

import matplotlib.pyplot as plt

def render_histogram_torch(A, dist):
    n, bins, patches = plt.hist(np.asarray(A.storage()), len(dist))
    ax = plt.gca()
    ax.set_ylim([0, A.numel()])
    plt.xticks(range(0, len(dist)))

    plt.xlabel('values')
    plt.ylabel('number of samples')
    plt.title(f' values: {[f"{i}" for i in range(0, len(dist))]}, dist: {[f"{round(d, 2)}" for d in dist]}, samples: {A.numel()}', fontweight="bold")
    plt.show()

def render_histogram(A, dist):
    n, bins, patches = plt.hist(np.asarray(A), len(dist))
    ax = plt.gca()
    ax.set_ylim([0, A.size])
    plt.xticks(range(0, len(dist)))

    plt.xlabel('values')
    plt.ylabel('number of samples')
    plt.title(f' values: {[f"{i}" for i in range(0, len(dist))]}, dist: {[f"{round(d, 2)}" for d in dist]}, samples: {A.size}', fontweight="bold")
    plt.show()

render_histogram(X, probs)



#
# torch
#

import torch

dist = probs
print("dist: ", dist)
kk_samples = torch.randint(low=0, high=K, size=(N,))        # 0/1/2
kk_samples_flat = torch.reshape(kk_samples, (1, N))
dice2_samples = torch.rand(size=(N, ))                              # 0.xxx
dice2_samples_flat = torch.reshape(dice2_samples, (1, N))

# K = len(J)                            # 3
# kk = int(np.floor(npr.rand() * K))    # 0/1/2
# dice2 = npr.rand()                    # 0.xxx
# if dice2 < q[kk]:                     # q[0/1/2] = 0.yyy
#     return kk                         # kk = 0/1/2
# else:
#     return J[kk]                      # J[kk] = 0/1/2
A = kk_samples_flat.clone()
for i in range(K):
    tensor_q = torch.ones(size=(1, N)) * q[i]
    mask_override_dice2 = torch.where(dice2_samples_flat < tensor_q, True, False)
    A = torch.where(mask_override_dice2 == True, i, A)

render_histogram_torch(A, dist)
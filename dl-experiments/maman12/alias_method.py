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

kk_arr = [0, 1, 1, 2, 2, 1, 2, 2, 2, 1]
height_arr = [0.6284530449295839,
 0.7508719449781908,
 0.7041699036208704,
 0.7896240197856693,
 0.5067288555329683,
 0.641978371417618,
 0.47726097537358003,
 0.5516907834153594,
 0.7235518837358279,
 0.7187565292938165]

def alias_draw(J, q, nn):
    K  = len(J)

    # kk = int(np.floor(npr.rand()*K))    # Draw from the overall uniform mixture.
    # height = npr.rand()                 # small one, or choosing the associated larger one.
    # kk_arr.append(kk)
    # height_arr.append(height)
    kk = kk_arr[nn]
    height = height_arr[nn]

    if height < q[kk]:
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
probs = [0.19, 0.73, 0.08]
J = [1, 0, 1]
q = [0.89235835, 1.,         0.55670321]

print("J: ", J)
print("k: ", q)
print("kk_arr: ", kk_arr)
print("height_arr: ", height_arr)

# Generate variates.
X = np.zeros(N)
for nn in range(N):
    X[nn] = alias_draw(J, q, nn)

print("X: ", X)
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

# Jkk_samples = torch.randint(low=0, high=K, size=(N,))        # 0/1/2
# height_samples = torch.rand(size=(N,))                              # 0.xxx
Jkk_samples = torch.tensor(kk_arr)
height_samples = torch.tensor(height_arr)
Jkk_samples_flat = torch.reshape(Jkk_samples, (1, N))
height_samples_flat = torch.reshape(height_samples, (1, N))

# K = len(J)                            # 3
# kk = int(np.floor(npr.rand() * K))    # 0/1/2
# dice2 = npr.rand()                    # 0.xxx
# if dice2 < q[kk]:                     # q[0/1/2] = 0.yyy
#     return kk                         # kk = 0/1/2
# else:
#     return J[kk]                      # J[kk] = 0/1/2
A = Jkk_samples_flat.clone()
B = height_samples_flat.clone()
Al = A.long()
tJ = torch.tensor(J)
C = torch.tensor(tJ[Al])
tq = torch.tensor(q)
D = torch.tensor(tq[Al])
E = torch.where(B < D, A, C)
# for i in range(K):
#     mask_kk = torch.where(A == i, True, False)
#     mask_height = torch.where(height_samples_flat < q[i], True, False)
#     mask = mask_kk & mask_height
#     A = torch.where(mask, A, i)


render_histogram_torch(E, dist)
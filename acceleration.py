import numpy as np
import time
import functions as f
from copy import deepcopy

t0 = time.clock()  # start time

a = int(input("State number of probes: "))
b = int(input("State number of centroids: "))

a -= a % 32  # otherwise results on GPU may be wrong
b -= b % 32

k = int(input("Shift = 1e- ?? "))
shift = 10**(-k)  # algorithms ending condition

DATA = f.random_matrix(a, b)  # a x b matrix, with generated random probes
DATA2 = DATA  # copy input DATA, so calculations may be later compared

[CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS] = f.centroid_creation(DATA, b)
CENTROIDS2 = deepcopy(CENTROIDS)  # copy starting CENTROIDS, so calculations may be later compared
CENTROIDS_SHIFT2 = deepcopy(CENTROIDS_SHIFT)
PREVIOUS_CENTROIDS2 = deepcopy(PREVIOUS_CENTROIDS)
# f.stan(CENTROIDS, CENTROIDS_SHIFT, DATA, b)  # check actual state of centroids & probes
# f.stan(CENTROIDS2, CENTROIDS_SHIFT2, DATA2, b)  # check actual state of centroids & probes

t1 = time.clock()   # cpu start time

q = 0  # iterator
while np.ndarray.max(CENTROIDS_SHIFT) > shift:  # check if actual max centroid shift meets conditions
    q += 1
    print(" ------------------- ")
    print("CPU iteration: ", q)
    print("Max shift: ", round(np.ndarray.max(CENTROIDS_SHIFT), k))
    # CPU -------------------------------------------------------------
    [DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS] = \
        f.algorithm_cpu(DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS, a, b)

t2 = time.clock() - t1  # cpu time
print("---------------------------------------------------------------------------------------------------------------")
t3 = time.clock()  # gpu start time

q = 0  # iterator
while np.ndarray.max(CENTROIDS_SHIFT2) > shift:  # check if actual max centroid shift meets conditions
    q += 1
    print(" ------------------- ")
    print("GPU iteration: ", q)
    print("Max shift: ", round(np.ndarray.max(CENTROIDS_SHIFT), k))
    # WERSJA GPU_lepsza -------------------------------------------------------------
    [DATA2, CENTROIDS2, CENTROIDS_SHIFT2, PREVIOUS_CENTROIDS2] = \
        f.algorithm_gpu(DATA2, CENTROIDS2, CENTROIDS_SHIFT2, PREVIOUS_CENTROIDS2, a, b)

t4 = time.clock() - t3  # gpu time

print("Summary: ")
print("Number of probes: ", a)
print("Number of centroids: ", b)
print("Iterations: ", q)
print("CPU timing: ", round(t2, 2), "[s]")
print("GPU_more timing: ", round(t4, 2), "[s]")
print("Faster: ", round(t2/t4, 2), "times")
print("Difference in results: ", round(np.ndarray.sum(DATA2) - np.ndarray.sum(DATA), 2))

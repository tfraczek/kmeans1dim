import numpy as np
import time
import functions as f

t0 = time.clock()  # start time

a = int(input("State number of probes: "))
b = int(input("State number of centroids: "))

a -= a % 32  # otherwise results on GPU may be wrong
b -= b % 32

k = int(input("Shift = 1e- ?? "))  # algorithms ending condition
shift = 10**(-k)
gpu_vs_cpu = input("Use gpu or cpu version of algorithm? ")  # decide to run algorithm on cpu/gpu

DATA = f.random_matrix(a, b)  # a x b matrix, with generated random probes

[CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS] = f.centroid_creation(DATA, b)  # create centroids

# f.stan(CENTROIDS, CENTROIDS_SHIFT, DATA, b)  # check actual state of centroids & probes

q = 0  # iterator
while np.ndarray.max(CENTROIDS_SHIFT) > shift:  # check if actual max centroid shift meets conditions
    q += 1
    print(" ------------------- ")
    print("Iteration: ", q)
    print("Max shift: ", round(np.ndarray.max(CENTROIDS_SHIFT), k))
    if gpu_vs_cpu == "gpu_part":
        # partial GPU -------------------------------------------------------------
        [DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS] = \
            f.algorithm_gpu_part(DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS, a, b)
    elif gpu_vs_cpu == "cpu":
        # CPU ---------------------------------------------------------------------
        [DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS] = \
            f.algorithm_cpu(DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS, a, b)
    elif gpu_vs_cpu == "gpu":
        # GPU ----------------------------------------------------------------------
        [DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS] = \
            f.algorithm_gpu(DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS, a, b)
    else:
        print("State gpu or cpu")
        quit()
    # f.stan(CENTROIDS, CENTROIDS_SHIFT, DATA, b)  # check actual state of centroids & probes

f.results(DATA, CENTROIDS, CENTROIDS_SHIFT, b, q, t0, shift)  # show the results of algorithm

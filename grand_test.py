import numpy as np
import time
import functions as f
from copy import deepcopy
import datetime

now = datetime.datetime.now()
name = "results_of_acceleration"+str(datetime.date.today())+".txt"
file = open(name, "w")
now = "Test date: " + str(now) + '\n'
file.write(now)
file.close()

max_probes = int(input("State the max n (there will be 32*n*n*n probes in final calculation "))
k = int(input("Shift = 1e- ?? "))

for n in range(max_probes):
    t0 = time.clock()  # start time
    a = 32 * (n+1)**3  # to lower the number of calculations
    b = 32

    shift = 10**(-k)

    DATA = f.random_matrix(a, b)  # a x b matrix, with generated random probes
    DATA2 = DATA  # copy input DATA, so calculations may be later compared

    [CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS] = f.centroid_creation(DATA, b)
    CENTROIDS2 = deepcopy(CENTROIDS)  # copy starting CENTROIDS, so calculations may be later compared
    CENTROIDS_SHIFT2 = deepcopy(CENTROIDS_SHIFT)
    PREVIOUS_CENTROIDS2 = deepcopy(PREVIOUS_CENTROIDS)
    # f.stan(CENTROIDS, CENTROIDS_SHIFT, DATA, b)

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
    print("-----------------------------------------------------------------------------------------------------------")
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

    file = open(name, "a")
    file.writelines("------------------------------------------------------------------------------------------" + '\n')
    file.writelines("Summary: ")
    string = "Number of probes: " + str(a) + '\n'
    file.writelines(string)
    string = "Number of centroids: " + str(b) + '\n'
    file.writelines(string)
    string = "Iterations: " + str(q) + '\n'
    file.writelines(string)
    string = "CPU timing: " + str(round(t2, 2)) + "[s]" + '\n'
    file.writelines(string)
    string = "GPU_more timing: " + str(round(t4, 2)) + "[s]" + '\n'
    file.writelines(string)
    string = "Faster: " + str(round(t2/t4, 2)) + " times" + '\n'
    file.writelines(string)
    string = "Difference in results: " + str(round(np.ndarray.sum(DATA2) - np.ndarray.sum(DATA), 2)) + '\n'
    file.writelines(string)
    string = "User set shift = " + str(shift)
    file.writelines(string)
    file.close()

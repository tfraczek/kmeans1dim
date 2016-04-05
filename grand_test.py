import numpy as np
import time
import functions as f
from copy import deepcopy
import datetime

now = datetime.datetime.now()
name = "wyniki_przyspieszenia"+str(datetime.date.today())+".txt"
file = open(name, "w")
now = "Test date: " + str(now) + '\n'
file.write(now)
file.close()


for n in range(1, 5):
    t0 = time.clock()  # czas początkowy
    a = 32 * n * n * n
    b = 32

    k = 7  # przesunięcie, poniżej którego nastąpi zakończenie działania algorytmu, dokładniej 1e-k
    # k = input("Shift = 1e- ??")
    shift = 10**(-k)

    DATA = f.random_matrix(a, b)
    DATA2 = DATA

    [CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS] = f.centroid_creation(DATA, b)

    CENTROIDS2 = deepcopy(CENTROIDS)

    CENTROIDS_SHIFT2 = deepcopy(CENTROIDS_SHIFT)

    PREVIOUS_CENTROIDS2 = deepcopy(PREVIOUS_CENTROIDS)
    # f.stan(CENTROIDS, CENTROIDS_SHIFT, DATA, b)

    t1 = time.clock()   # gpu start time

    # pętelka
    q = 0  # liczba iteracji
    t = 1  # może być też max z CENTROIDS_SHIFT ale to i tak jest 1
    while t > shift:
        q += 1
        print(" ------------------- ")
        print("CPU iteration: ", q)
        print("Max shift: ", round(t, 7))
        # WERSJA CPU -------------------------------------------------------------
        [DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS] = \
            f.algorithm_cpu(DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS, a, b)
        t = np.ndarray.max(CENTROIDS_SHIFT)

    t2 = time.clock() - t1  # gpu time
    print("-----------------------------------------------------------------------------------------------------------")
    t3 = time.clock()  # cpu start time
    # pętelka
    q = 0  # liczba iteracji
    t = 1  # może być też max z CENTROIDS_SHIFT ale to i tak jest 1
    while t > shift:
        q += 1
        print(" ------------------- ")
        print("GPU iteration: ", q)
        print("Max shift: ", round(t, 7))
        # WERSJA GPU_lepsza -------------------------------------------------------------
        [DATA2, CENTROIDS2, CENTROIDS_SHIFT2, PREVIOUS_CENTROIDS2] = \
            f.algorithm_gpu(DATA2, CENTROIDS2, CENTROIDS_SHIFT2, PREVIOUS_CENTROIDS2, a, b)

        t = np.ndarray.max(CENTROIDS_SHIFT2)

    t4 = time.clock() - t3  # cpu time
    print("-----------------------------------------------------------------------------------------------------------")
    print("Summary: ")
    print("Number of probes: ", a)
    print("Number of centroids: ", b)
    print("Iterations: ", q)
    print("CPU timing: ", round(t2, 2), "[s]")
    print("GPU_more timing: ", round(t4, 2), "[s]")
    print("Faster: ", round(t2/t4, 2), "times")
    print("Difference in results: ", round(np.ndarray.sum(DATA2) - np.ndarray.sum(DATA), 2))
    file = open(name, "a")
    file.writelines("------------------------------------------------------------------------------ -----------" + '\n')
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

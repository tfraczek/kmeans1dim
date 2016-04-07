import numpy as np
import time
import functions as f
from copy import deepcopy

t0 = time.clock()  # czas początkowy

a = int(input("Number of probes == "))
b = int(input("Number of centroids == "))

a -= a % 32  # przy braku podzielności przez 32 może zaistnieć błąd w wynikach
b -= b % 32

# k = 7  # przesunięcie, poniżej którego nastąpi zakończenie działania algorytmu, dokładniej 1e-k
k = input("Shift = 1e- ??")
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
t = 1  # 1, aby wykonać pierwszą iterację
while t > shift:
    q += 1
    print(" ------------------- ")
    print("CPU iteration: ", q)
    print("Max shift: ", round(t, 7))
    # WERSJA CPU -------------------------------------------------------------
    [DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS] = \
        f.algorithm_cpu(DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS, a, b)
    t = np.ndarray.max(CENTROIDS_SHIFT)

t2 = time.clock() - t1  # cpu time
print("---------------------------------------------------------------------------------------------------------------")
t3 = time.clock()  # gpu start time
# pętelka
q = 0  # liczba iteracji
t = 1  # 1, aby wykonać pierwszą iterację
while t > shift:
    q += 1
    print(" ------------------- ")
    print("GPU iteration: ", q)
    print("Max shift: ", round(t, 7))
    # WERSJA GPU_lepsza -------------------------------------------------------------
    [DATA2, CENTROIDS2, CENTROIDS_SHIFT2, PREVIOUS_CENTROIDS2] = \
        f.algorithm_gpu(DATA2, CENTROIDS2, CENTROIDS_SHIFT2, PREVIOUS_CENTROIDS2, a, b)

    t = np.ndarray.max(CENTROIDS_SHIFT2)

t4 = time.clock() - t3  # gpu time
print("Summary: ")
print("Number of probes: ", a)
print("Number of centroids: ", b)
print("Iterations: ", q)
print("CPU timing: ", round(t2, 2), "[s]")
print("GPU timing: ", round(t4, 2), "[s]")
print("Faster: ", round(t2/t4, 2), "times")
print("Difference in results: ", round(np.ndarray.sum(DATA2) - np.ndarray.sum(DATA), 2))

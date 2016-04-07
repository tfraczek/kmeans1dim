import numpy as np
import time
import functions as f

t0 = time.clock()  # czas początkowy

a = int(input("Number of probes == "))
b = int(input("Number of centroids == "))

a -= a % 32  # przy braku podzielności przez 32 może zaistnieć błąd w wynikach
b -= b % 32

# k = 5  # przesunięcie, poniżej którego nastąpi zakończenie działania algorytmu, dokładniej 1e-k
k = int(input("Shift = 1e- ?? "))
shift = 10**(-k)
gpu_vs_cpu = "cpu"  # run gpu or cpu version of main algorithm
# gpu_vs_cpu = input("Use gpu or cpu version of algorithm")

DATA = f.random_matrix(a, b)
[CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS] = f.centroid_creation(DATA, b)

# f.stan(CENTROIDS, CENTROIDS_SHIFT, DATA, b)

# pętelka
q = 0  # liczba iteracji
while np.ndarray.max(CENTROIDS_SHIFT) > shift:
    q += 1
    print(" ------------------- ")
    print("Iteration: ", q)
    print("Max shift: ", round(np.ndarray.max(CENTROIDS_SHIFT), k))
    if gpu_vs_cpu == "gpu_part":
        # WERSJA GPU -------------------------------------------------------------
        [DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS] = \
            f.algorithm_gpu_part(DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS, a, b)
    elif gpu_vs_cpu == "cpu":
        # WERSJA CPU -------------------------------------------------------------
        [DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS] = \
            f.algorithm_cpu(DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS, a, b)
    elif gpu_vs_cpu == "gpu":
        # WERSJA GPU lepsza ------------------------------------------------------
        [DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS] = \
            f.algorithm_gpu(DATA, CENTROIDS, CENTROIDS_SHIFT, PREVIOUS_CENTROIDS, a, b)
    else:
        print("State gpu or cpu")
        quit()
    # f.stan(CENTROIDS, CENTROIDS_SHIFT, DATA, b)

f.results(DATA, CENTROIDS, CENTROIDS_SHIFT, b, q, t0, shift)  # przedstawienie wyników

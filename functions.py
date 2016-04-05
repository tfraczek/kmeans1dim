import numpy as np
from pycuda import compiler, gpuarray
# noinspection PyUnresolvedReferences
import pycuda.autoinit
import time


# tworzenie tablicy z losowymi danymi ---------------------------------------------------------------------------------
# wiersz 0 : nasze dane
# wiersz 1 do b : odległość od centroidów
# wiersz b+1 : do którego centroidu należy
def random_matrix(a, b):
    data = np.zeros(shape=(b+2, a), dtype=np.float64)
    data = np.append(np.random.randn(1, a), data, 0)
    return data


# tworzenie tablicy z centroidami -------------------------------------------------------------------------------------
# centroidy będą to początkowo najmniejsza oraz największa oraz dodatkowo wszystkie pomiędzy nimi z równymi odstępami
def centroid_creation(data, b):
    c = np.ndarray.max(data[0])
    d = np.ndarray.min(data[0])
    e = c - d
    centroids = [c, d]
    for i in range(b-2):
        centroids = np.insert(centroids, 1, np.float64((d+(e/(b-1))+i*(e/(b-1)))))
    centroids_shift = np.ones_like(centroids)
    previous_centroids = np.ones_like(centroids)
    for k in range(len(centroids)):
        previous_centroids[k] = centroids[k]
    return centroids, centroids_shift, previous_centroids


# sprawdzenie stanu centroidów ----------------------------------------------------------------------------------------
def stan(centroids, centroids_shift, data, b):
    print("Centroids:        -------------------")
    print(centroids)
    print("Shift:            -------------------")
    print(centroids_shift)
    print("DATA in centroid: -------------------")
    print(data[b+1])
    input("Press to continue")


# obliczanie odległości od centroidów na GPU --------------------------------------------------------------------------
def algorithm_gpu_part(data, centroids, centroids_shift, previous_centroids, a, b):
    gpu_data = gpuarray.to_gpu(data)
    gpu_centroids = gpuarray.to_gpu(centroids)

    kernel_code_distances = """
    __global__ void distanceKernel(double *a, double *b, double *c, int d)
    {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int bw = blockDim.x;
        int bh = blockDim.y;
        int x = tx + bx*bw;
        int y = ty + by*bh;
        a[(y + d*x )] = abs(b[x] - c[y]);
    }
    """
    # a[ty+d*tx] = abs(b[tx] - c[ty]);
    mod = compiler.SourceModule(kernel_code_distances)
    distance = mod.get_function("distanceKernel")

    if a > 32:
        a3 = (a-(a % 32))/32
        a2 = 32
    else:
        a2 = a
        a3 = 1

    if b > 32:
        b3 = (b-(b % 32))/32
        b2 = 32
    else:
        b2 = b
        b3 = 1

    distance(
        gpu_data[1:len(gpu_data), :], gpu_centroids, gpu_data[0], np.int32(a),
        block=(int(b2), int(a2), 1),
        grid=(int(b3), int(a3))
    )

    gpu_data = gpu_data.get()
    data = gpu_data
    for j in range(a):
        data[b+2, j] = np.ndarray.max(data[0])
        for i in range(b):
            if data[i+1, j] < data[b+2, j]:
                data[b+1, j] = i
                data[b+2, j] = data[i+1, j]
    z = 0  # ilość próbek w danym centroidzie
    for k in range(len(centroids)):
        centroids[k] = 0
        for j in range(a):
            if k == data[b+1, j]:
                centroids[k] += data[0, j]
                z += 1
        if z > 0:  # na wypadek gdyby w centroidzie nie było próbek
            centroids[k] /= z
            z = 0
        centroids_shift[k] = abs(previous_centroids[k] - centroids[k])
        previous_centroids[k] = centroids[k]
    return data, centroids, centroids_shift, previous_centroids


# obliczanie na GPU: odległości od centroidów oraz wybór najlepszego centroidu dla danej próbki -----------------------
def algorithm_gpu(data, centroids, centroids_shift, previous_centroids, a, b):
    gpu_data = gpuarray.to_gpu(data)
    gpu_centroids = gpuarray.to_gpu(centroids)

    kernel_code_distances = """
    __global__ void distanceKernel(double *a, double *b, double *c, int d)
    {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int bw = blockDim.x;
        int bh = blockDim.y;
        int x = tx + bx*bw;
        int y = ty + by*bh;

        a[(y + d*x )] = abs(b[x] - c[y]);
    }
    """

    mod = compiler.SourceModule(kernel_code_distances)
    distance = mod.get_function("distanceKernel")

    if a > 32:
        # a3 = (a-(a % 32))/32
        a3 = a/32
        a2 = 32
    else:
        a2 = a
        a3 = 1

    if b > 32:
        b3 = b/32
        b2 = 32
    else:
        b2 = b
        b3 = 1

    distance(
        gpu_data[1:len(gpu_data), :], gpu_centroids, gpu_data[0], np.int32(a),
        block=(int(b2), int(a2), 1),
        grid=(int(b3), int(a3))
    )

    gpu_data = gpu_data.get()
    data = gpu_data

    kernel_code_best_centroid = """
    __global__ void BestCentroidKernel(double *tab, double max, int b, int a)
    {
        int tx = threadIdx.x;
        int bx = blockIdx.x;
        int bw = blockDim.x;
        int x = tx + bx*bw;

        tab[x + a*(b+2)] = max;
        for(int n = 0; n < b; n++){
            if( tab[a*(n+1)+x] < tab[x+a*(b+2)] ){
                tab[a*(b+1)+x] = n;
                tab[a*(b+2)+x] = tab[x+a*(n+1)];
            }
        }
    }
    """

    gpu_data = gpuarray.to_gpu(data)

    mod = compiler.SourceModule(kernel_code_best_centroid)
    best_centroid = mod.get_function("BestCentroidKernel")

    best_centroid(
        gpu_data[:, :], np.ndarray.max(data[0]), np.int32(b), np.int32(a),
        block=(int(a2), 1, 1),
        grid=(int(a3), 1)
    )

    gpu_data = gpu_data.get()
    data = gpu_data

    z = 0  # ilość próbek w danym centroidzie
    for k in range(len(centroids)):
        centroids[k] = 0
        for j in range(a):
            if k == data[b+1, j]:
                centroids[k] += data[0, j]
                z += 1
        if z > 0:  # na wypadek gdyby w centroidzie nie było próbek
            centroids[k] /= z
            z = 0
        centroids_shift[k] = abs(previous_centroids[k] - centroids[k])
        previous_centroids[k] = centroids[k]
    return data, centroids, centroids_shift, previous_centroids


# algorytm w pełni zaimplementowany na CPU ----------------------------------------------------------------------------
def algorithm_cpu(data, centroids, centroids_shift, previous_centroids, a, b):
    for i in range(b):
        for j in range(a):
            data[i+1, j] = abs(centroids[i] - data[0, j])
    for j in range(a):
        data[b+2, j] = np.ndarray.max(data[0])
        for i in range(b):
            if data[i+1, j] < data[b+2, j]:
                data[b+1, j] = i
                data[b+2, j] = data[i+1, j]
    z = 0  # ilość próbek w danym centroidzie
    for k in range(len(centroids)):
        centroids[k] = 0
        for j in range(a):
            if k == data[b+1, j]:
                centroids[k] += data[0, j]
                z += 1
        if z > 0:  # na wypadek gdyby w centroidzie nie było próbek
            centroids[k] /= z
            z = 0
        centroids_shift[k] = abs(previous_centroids[k] - centroids[k])
        previous_centroids[k] = centroids[k]
    return data, centroids, centroids_shift, previous_centroids


# wyświetlenie wyników: dane wejściowe, centroidy, ostatnie przesunięcie centroidów, ilość iteracji, czas wykonania ---
def results(data, centroids, centroids_shift, b, q, t0, shift):
    print(" ")
    print("RESULTS:              -----------------")
    print("input samples:        -----------------")
    print(data[0])
    print("centroids:            -----------------")
    print(centroids)
    print("centroids last shift: -----------------")
    print(centroids_shift)
    print("User set shift        -----------------")
    print(shift)
    print("centroids of samples: -----------------")
    print(data[b+1])
    print(" ")
    print(" Number of iterations: ", q)
    print(" Total time:  ", round(time.clock() - t0, 2), "[s]")

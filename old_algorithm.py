import numpy as np
import time
from pycuda import compiler, gpuarray
# noinspection PyUnresolvedReferences
import pycuda.autoinit

t0 = time.clock()  # czas początkowy

a = 64  # ilosc probek musi być wiwelokronością 32 ( problem z macierzą dwuwymiarową... )
b = 32  # ilosc centroidow tak samo ( ale uwaga, może być mniej niż 32)

# tworzenie tablicy z danymi ------------------------------------------------------------------------------------------
# wiersz 0 : nasze dane
# wiersz 1-b : odległość od centroidów
# wiersz b+1 : do którego centroidu należy
DATA = np.zeros(shape=(b+2, a), dtype=np.float64)
DATA = np.append(np.random.randn(1, a), DATA, 0)
# tworzenie tablicy z centroidami -------------------------------------------------------------------------------------
# rozrzut danych:
c = np.ndarray.max(DATA[0])
d = np.ndarray.min(DATA[0])
e = c - d
# centroidy będą to początkowo najmniejsza oraz najwieszka oraz dodatkowo wszystkie pomiedzy nimi z równymi odstępami
CENTROIDS = [c, d]
for i in range(b-2):
    CENTROIDS = np.insert(CENTROIDS, 1,  np.float64(d+(e/(b-1))+i*(e/(b-1))))
# przesuniecie centroidu
CENTROIDS_SHIFT = np.ones_like(CENTROIDS)
OLD_CENTROIDS = np.ones_like(CENTROIDS)
for k in range(len(CENTROIDS)):
    OLD_CENTROIDS[k] = CENTROIDS[k]
#  sprawdzenie stanu początkowego -------------------------------------------------------------------------------------
print("CENTROIDY POCZĄTKOWE: -------------")
print(CENTROIDS)
# print("przesunięcie: -------------------")
# print(CENTROIDS_SHIFT)
# print("DANE w centr: -------------------")
# print(DATA[b+1])
# input("Naciśnij, by kontynuować")
#  pętelka
q = 0
t = 1  # może być też max z CENTROIDS_SHIFT ale to i tak jest 1
t_sum = 0  # suma róznicy czasu
i_s = 0  # suma iteracji
while t > 0.000001:
    q += 1
    print(" ------------------- ")
    print("Iteracja numer: ", q)
    print("Max z przesunięć: ", round(t, 7))
    t1 = time.clock()
    # WERSJA GPU -------------------------------------------------------------
    GPU_DATA = gpuarray.to_gpu(DATA)
    GPU_CENTROIDS = gpuarray.to_gpu(CENTROIDS)

    kernel_code_template = """
    __global__ void MatrixMulKernel(double *a, double *b, double *c, int d)
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
    mod = compiler.SourceModule(kernel_code_template)
    matrixmul = mod.get_function("MatrixMulKernel")

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

    matrixmul(
        GPU_DATA[1:len(GPU_DATA), :], GPU_CENTROIDS, GPU_DATA[0], np.int32(a),
        block=(int(b2), int(a2), 1),
        grid=(int(b3), int(a3))
    )

    GPU_DATA = GPU_DATA.get()
    # DATA = GPU_DATA
    t2 = time.clock()
    # WERSJA CPU -------------------------------------------------------------
    for i in range(b):
        for j in range(a):
            DATA[i+1, j] = abs(CENTROIDS[i] - DATA[0, j])
    t3 = time.clock()
    # print(DATA)
    # print(GPU_DATA)
    # print(GPU_DATA - DATA)
    print("Różnica czasu to: ", round((t3 - t2) - (t2 - t1), 2))
    print("Różnica pomiędzy GPU a CPU to : ", round(np.ndarray.sum(DATA) - np.ndarray.sum(GPU_DATA), 2))
    # quit()
    # input("Naciśnij, by kontynuować")
    # KONIEC RÓŻNYCH WERSJI --------------------------------------------------
    # print(time.clock())
    t_sum += (t3 - t2) - (t2 - t1)
    for j in range(a):
        DATA[b+2, j] = c
        for i in range(b):
            if DATA[i+1, j] < DATA[b+2, j]:
                DATA[b+1, j] = i
                DATA[b+2, j] = DATA[i+1, j]
    z = 0  # ilość próbek w danym centroidzie
    #  checkpoint 2
    # print(DATA)
    # print(CENTROIDS)
    # print(time.clock())
    for k in range(len(CENTROIDS)):
        CENTROIDS[k] = 0
        for j in range(a):
            if k == DATA[b+1, j]:
                CENTROIDS[k] += DATA[0, j]
                z += 1
        if z > 0:  # na wypadek gdyby w centroidze nie było próbek
            CENTROIDS[k] /= z
            z = 0
        CENTROIDS_SHIFT[k] = abs(OLD_CENTROIDS[k] - CENTROIDS[k])
        OLD_CENTROIDS[k] = CENTROIDS[k]
        # print("CENTROIDY: ----------------------")
        # print(CENTROIDS)
        # print("przesunięcie: ---------------------")
        # print(CENTROIDS_SHIFT)
        # print("DANE w centroidzie: ---------------")
        # print(DATA[b+1])
        # input("Naciśnij, by kontynuować")
    t = np.ndarray.max(CENTROIDS_SHIFT)
    i_s += 1
    # print(time.clock())
    # input("Naciśnij, by kontynuować")

print(" ")
print("WYNIKI: ---------------------------")
print("DANE WEJŚCIOWE:   -----------------")
print(DATA[0])
print("CENTROIDY: ------------------------")
print(CENTROIDS)
print("przesunięcie: ---------------------")
print(CENTROIDS_SHIFT)
print("DANE w centroidzie: ---------------")
print(DATA[b+1])
# input("Naciśnij, by kontynuować")
print(" ")
print(" Podsumowanie: --------------------")
print(" W sumie iteracji: ", i_s)
print(" W sumie, GPU wykonywało obliczenia szybciej o: ", round(t_sum, 2))
print(" Całkowity czas:  ", round(time.clock() - t0, 2))

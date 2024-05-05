import numpy as np
import matplotlib.pyplot as plt

file_path = "zad3.txt"
data = np.loadtxt(file_path)
X = data[:, 0]
Y = data[:, 2]


def wspolczynnik(macierzY, h):
    n = macierzY.shape[0]
    macierzB = np.zeros([n, n])
    macierzB[0][0] = - 3.0 / h
    macierzB[0][2] = 3.0 / h

    macierzB[n - 1][n - 1] = 3.0 / h
    macierzB[n - 1][n - 3] = -3.0 / h

    for i in range(1, n - 1):
        macierzB[i][i] = 4.0
        macierzB[i][i - 1] = 1.0
        macierzB[i][i + 1] = 1.0

    return macierzB


def B(z, h):
    if z >= -2 * h and z <= -h:
        return 1.0 / h ** 3 * (z + 2 * h) ** 3
    elif z >= -h and z <= 0:
        return 1.0 / h ** 3 * (h ** 3 + 3 * h ** 2 * (z + h) + 3 * h * (z + h) ** 2 - 3 * (z + h) ** 3)
    elif z >= 0 and z <= h:
        return 1.0 / h ** 3 * (h ** 3 + 3 * h ** 2 * (h - z) + 3 * h * (h - z) ** 2 - 3 * (h - z) ** 3)
    elif z >= h and z <= 2 * h:
        return 1.0 / h ** 3 * (2 * h - z) ** 3
    else:
        return 0


def wyznacz_wartosc(macierzK, x2, h, x):
    n = x2.shape[0]
    sum = 0

    for i in range(n):
        z = x - x2[i]
        sum += macierzK[i] * B(z=z, h=h)
    return sum
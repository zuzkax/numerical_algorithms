import numpy as np
import matplotlib.pyplot as plt


def pochodna(y, h):
    p = np.zeros([len(y)])

    p[0] = (y[1] - y[0]) / h

    for i in range(1, len(y) - 1):
        p[i] = (y[i + 1] - y[i - 1]) / (2 * h)

    p[len(y) - 1] = (y[len(y) - 1] - y[len(y) - 2]) / h

    return p



data = np.loadtxt('siatka0.txt')

X = data[:, 0]
Y = data[:, 2]

a = np.zeros([len(X)])

h = 0.1

p = pochodna(Y, h)


plt.plot(X, p, label="pochodna cząstkowa")
plt.title("pochodna cząstkowa dla y = 0.0")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

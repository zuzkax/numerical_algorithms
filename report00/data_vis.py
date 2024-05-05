import numpy as np
import matplotlib.pyplot as plt

file_path = "139374.dat"
data = np.loadtxt(file_path)

x = data[:, 0]
y = data[:, 1]
f = data[:, 2]

def zad1(x, y, f):
    y_const = np.unique(y)

    for i in y_const:
        listx = []
        listf = []
        index = 0
        while index < len(y):
            if y[index] == i:
                listx.append(x[index])
                listf.append(f[index])
            index += 1
        plt.scatter(listx, listf, label=f"y = {i}")

    plt.xlabel('x')
    plt.ylabel('F(x, y)')
    plt.title('Wykres F(x, y) dla każdej linii y=const')
    plt.legend()
    plt.show()

zad1(x, y, f)







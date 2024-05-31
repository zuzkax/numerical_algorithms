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

h = 0.1
p = pochodna(Y, h)

przedzialy = []

i = 0
while i < len(p) - 1:
    znak = np.sign(p[i])
    j = i + 1
    while j < len(p) and np.sign(p[j]) == znak:
        j += 1
    przedzialy.append((i, j - 1, znak))
    i = j

for przedzial in przedzialy:
    start, end, znak = przedzial
    if znak > 0:
        print(f"W przedziale [{X[start]}, {X[end+1]}], funkcja jest rosnąca.")
    elif znak < 0:
        print(f"W przedziale [{X[start]}, {X[end + 1]}], funkcja jest malejąca.")
    else:
        print(f"W przedziale [{X[start]}, {X[end]}], funkcja jest stała.")


plt.fill_between(X, 0, p, where=(p >= 0), color='green', alpha=0.3)
plt.fill_between(X, 0, p, where=(p < 0), color='red', alpha=0.3)


plt.plot(X,p, label = 'F\'(x,y)')
plt.plot(X,Y,label = 'F(x,y)')
plt.title('Monotoniczność')
plt.legend()
plt.show()

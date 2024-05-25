import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('siatka5.txt')

def LagrangeCoeff(X, Y):
    n = np.size(X)
    a = np.zeros(n)
    for i in range(n):
        m = 1
        for j in range(n):
            if j != i:
                m = m * (X[i] - X[j])
        a[i] = Y[i] / m
    return a

def LagrangeVal(macierzWsp, macierzX, x):
    n = macierzWsp.shape[0]
    y = 0
    for i in range(n):
        m = 1
        for j in range(n):
            if j != i:
                m = m * (x - macierzX[j])
        y = y + m * macierzWsp[i]
    return y


def interpolacja_przedzialami(X, Y, segment_size=5):
    n = len(X)
    xx = []
    yy = []

    for i in range(0, n, segment_size - 1):
        end = i + segment_size
        if end > n:
            end = n
        X_segment = X[i:end]
        Y_segment = Y[i:end]
        a = LagrangeCoeff(X_segment, Y_segment)

        xx_segment = np.linspace(X_segment[0], X_segment[-1], 100)
        yy_segment = LagrangeVal(macierzWsp=a, macierzX=X_segment, x=xx_segment)

        xx.extend(xx_segment)
        yy.extend(yy_segment)

        if end == n:
            break

    return np.array(xx), np.array(yy)

X = data[:, 0]
Y = data[:, 2]


xx, yy = interpolacja_przedzialami(X, Y)

plt.plot(xx, yy, label='Interpolacja wielomianowa')
plt.scatter(X, Y, color='green', label='Węzły')
plt.title("Interpolacja wielomianowa dla y = 0.5")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

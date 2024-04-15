import numpy as np
import matplotlib.pyplot as plt

def LagrangeCoeff(macierzX, macierzY):
    n = macierzX.shape[0]
    macierzWsp = np.zeros(n)

    for i in range(n):
        macierzWsp[i] = 1.0
        for j in range(n):
            if i != j:
                macierzWsp[i] *= 1.0 /(macierzX[i] - macierzX[j])

        macierzWsp[i] = macierzY[i] * macierzWsp[i]
    return macierzWsp


def LagrangeVal(macierzWsp, macierzX, x):
    n = macierzWsp.shape[0]
    m = 1
    y = 0

    for i in range(n):
        m = 1
        for j in range(n):
            if j != i:
                m = m * (x - macierzX[j])
        y = y + m * macierzWsp[i]
    return y

def f(x):
    return 1/(1+25 * x**2)

def generate_points(n):
    return np.linspace(-1, 1, n)

l = input("podaj liczbe punktow")
n = int(l)

macierzX = generate_points(n)
macierzY = np.zeros([n])

for i in range(n):
    macierzY[i] = f(macierzX[i])

print(macierzX)
print(macierzY)
macierzWsp = LagrangeCoeff(macierzX, macierzY)

yy = np.zeros([100])
xx = np.linspace(-1.0,1.0,100)

for i in range(100):
    yy[i] = LagrangeVal(macierzWsp, macierzX, xx[i])


plt.plot(xx, yy, label = 'wielomian Lagrange\'a')
plt.scatter(macierzX, macierzY, color= 'red', label = 'wezly')
plt.title(f"Wielomian Lagrange\'a dla {n} punktow")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

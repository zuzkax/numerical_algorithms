import numpy as np
import matplotlib.pyplot as plt

#przykladowe punkty
macierzX = np.array([-0.88, -0.549, 0.44, 0.487, 0.99])
macierzY = np.random.rand(5)


def LagrangeCoeff(macierzX, macierzY): #wyznacza wspolczynniki wilomianu
    n = macierzX.shape[0]
    macierzWsp = np.zeros(n)

    for i in range(n):
        macierzWsp[i] = 1.0
        for j in range(n):
            if i != j:
                macierzWsp[i] *= 1.0 /(macierzX[i] - macierzX[j])

        macierzWsp[i] = macierzY[i] * macierzWsp[i]
    return macierzWsp



def LagrangeVal(macierzWsp, macierzX, x):#wyznacza wartosci wielomianu
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


macierzWsp = LagrangeCoeff(macierzX=macierzX, macierzY=macierzY)

#rysowanie wykresu
xx = np.linspace(-1,1,100)
yy = LagrangeVal(macierzWsp=macierzWsp, macierzX=macierzX, x=xx)

plt.plot(xx, yy, label = 'wielomian')
plt.scatter(macierzX, macierzY, color= 'red', label = 'wezly')
plt.title("Wielomian Lagrange\'a")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
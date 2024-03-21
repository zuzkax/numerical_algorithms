import numpy as np
import matplotlib.pyplot as plt

#przykladowe punkty
macierzX = np.array([-8.053, -7.849, -5.44, -1.487,2.170])
macierzY = np.array([-4.309, 0.786, 2.96, 3.312, 4.595])

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
xx = np.linspace(-10,3,100)
yy = LagrangeVal(macierzWsp=macierzWsp, macierzX=macierzX, x=xx)

plt.plot(xx, yy, label = 'wielomian')
plt.scatter(macierzX, macierzY, color= 'red', label = 'wezly')
plt.title("Wielomian Lagrange\'a")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

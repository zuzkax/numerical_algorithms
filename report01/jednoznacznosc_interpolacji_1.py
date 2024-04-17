import numpy as np
import matplotlib.pyplot as plt

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

def wsp(x,y):
    n = len(x)
    A = np.zeros([n,n])

    #tworzenie macierzy glownej
    for i in range(n):
        for j in range(n):
            A[i,j] = x[i]**j

    #rozwiazanie ukladu rownan
    coef = np.linalg.solve(A,macierzY)
    return coef

def fun_bazowa(x,k):
    return x**k

def suma(a,x):
    sum = 0
    n = len(a)
    for i in range(n):
        sum += a[i] * fun_bazowa(x,i)
    return sum

#przykladowe punkty
macierzX = np.array([-0.88, -0.549, 0.44, 0.487, 0.99])
macierzY = np.random.rand(5)

macierzWsp = LagrangeCoeff(macierzX, macierzY)
A = wsp(macierzX,macierzY)

xx_l = np.linspace(-1,1,100)
yy_l = LagrangeVal(macierzWsp, macierzX, xx_l)

xx_w = np.linspace(-1,1,100)
yy_w = np.zeros([100])
for i in range(100):
    yy_w[i] = suma(A,xx_w[i])


plt.plot(xx_w, yy_w, label = 'interpolacja wielomianowa')

plt.plot(xx_l, yy_l, label = "wielomian Lagrange\'a")
plt.scatter(macierzX, macierzY, color= 'red', label = 'wezly')
plt.title("Wielomian Lagrange\'a")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

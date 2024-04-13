import matplotlib.pyplot as plt
import numpy as np

def wspolczynnik(macierzY, h):
    n = macierzY.shape[0]
    macierzB = np.zeros([n,n])
    macierzB[0][0] = - 3.0/h
    macierzB[0][2] = 3.0/h

    macierzB[n-1][n-1] = 3.0/h 
    macierzB[n-1][n-3] = -3.0/h
    
    for i in range(1,n-1):
        macierzB[i][i] = 4.0
        macierzB[i][i-1] = 1.0
        macierzB[i][i+1] = 1.0

    return macierzB


def B(z,h):
    if z >= -2*h and z <= -h:
        return 1.0/h**3*(z+2*h)**3
    elif z >= -h and z <= 0:
        return 1.0/h**3*(h**3 + 3*h**2*(z+h) + 3*h*(z+h)**2 - 3*(z+h)**3)
    elif z >= 0 and z <= h:
        return 1.0/h**3*(h**3 + 3*h**2*(h-z) + 3*h*(h-z)**2 - 3*(h-z)**3)
    elif z >= h and z <= 2*h:
        return 1.0/h**3*(2*h - z)**3
    else:
        return 0


def wyznacz_wartosc(macierzK, x2,  h, x):
    n = x2.shape[0] 
    sum = 0

    for i in range(n):
        z = x - x2[i]
        sum += macierzK[i] * B(z=z, h=h)
    return sum


macierzX = np.array([-1.00, -0.60, -0.20, 0.20, 0.60, 1.00])
macierzY = np.array([1.00, 5.58, 7.03, 5.11, 0.84, 9.29, 1.89, 1.00])

h = macierzX[1] - macierzX[0]

macierzB = wspolczynnik(macierzY, h)
macierzK = np.linalg.solve(macierzB, macierzY)

x2 = np.concatenate(([macierzX[0]-h],macierzX,[macierzX[-1]+h]))

xx = np.linspace(-1.0,1.0,100)
yy = np.zeros([100])


for i in range(100):
    yy[i] = wyznacz_wartosc(macierzK, x2, h, xx[i])

plt.plot(xx,yy, label = "bsplajn")
plt.scatter(macierzX,macierzY[1:-1], color = "red", label = "punkty")
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.show()




import numpy as np
import matplotlib.pyplot as plt

def wsp(x,y):
    n = len(x)
    A = np.zeros([n,n])

    #tworzenie macierzy glownej
    for i in range(n):
        for j in range(n):
            A[i,j] = x[i]**j

    #rozwiazanie ukladu rownan
    coef = np.linalg.solve(A,Y)
    return coef

def fun_bazowa(x,k):
    return x**k

def suma(a,x):
    sum = 0
    n = len(a)
    for i in range(n):
        sum += a[i] * fun_bazowa(x,i)
    return sum

X = np.linspace(-2,2,6)
Y = np.random.rand(6)

A = wsp(X,Y)

xx = np.linspace(-3,3,100)
yy = np.zeros([100])

for i in range(100):
    yy[i] = suma(A,xx[i])

plt.plot(xx,yy)
plt.scatter(X, Y, color= 'red', label = 'wezly')
plt.title("Interpolacja wielomianowa")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
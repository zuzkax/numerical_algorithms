import numpy as np
import matplotlib.pyplot as plt

def wspolczynniki(m,X,Y):
    l = len(X)
    M = np.zeros([m+1, m+1])
    N = np.zeros([m+1])
    for i in range(m +1):
        for j in range(m + 1):
            for k in range(l):
                M[i][j] += X[k]**(i+j)

    for i in range(m+1):
        for k in range(l):
            N[i] += Y[k] * X[k] **  i

    a = np.linalg.solve(M,N)
    return a

def wielomian_aproksymujacy(x,a,m):
    w = 0
    for i in range(m+1):
        w += a[i] * x ** i
    return w

    

X = np.array([0.00,0.80,1.60,2.40,3.20, 4.00])
Y = np.array([0.71,2.28,4.49,7.17,10.27,16.60])


a1 = wspolczynniki(2,X,Y)
a2 = wspolczynniki(1,X,Y)

xx = np.linspace(0.0,5.00,100)
yy = np.zeros([100])
yyy = np.zeros([100])

for i in range(100):
    yy[i] = wielomian_aproksymujacy(xx[i],a1, 2)

for i in range(100):
    yyy[i] = wielomian_aproksymujacy(xx[i],a2, 1)

wsp = np.polyfit(X,Y,2)
wielomian = np.poly1d(wsp)

x_zakres = np.linspace(0.0,5.0,100)
#y_wiel = np.zeros
y_wiel = wielomian(x_zakres)

plt.plot(x_zakres,y_wiel, label = 'aproksymacja st 2, z fun wbudowanej')
plt.scatter(X,Y, color = 'red', label = 'punkty')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

plt.plot(xx,yy, label = 'aproksymacja st 2')
plt.plot(xx,yyy, label = 'aproksymacja st 1')
plt.scatter(X,Y, color = 'red', label = 'punkty')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

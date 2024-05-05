import numpy as np
import matplotlib.pyplot as plt


file_path = "zad3.txt"
data = np.loadtxt(file_path)
X = data[:, 0]
Y = data[:, 2]

def eliminacjaGaussa(macierzA, wektorPrawy):
    n = macierzA.shape[0]
    wektorNiewiadomych = np.zeros([n])

    for i in range(n):
        for j in range(i+1,n):
            m = macierzA[j][i]/ macierzA[i][i]
            for k in range(i,n):
                macierzA[j][k] -= m * macierzA[i][k]
            wektorPrawy[j] -= m * wektorPrawy[i]


    wektorNiewiadomych[n-1] = wektorPrawy[n-1]/ macierzA[n-1][n-1]

    for i in range(n-2, -1, -1):
        suma = 0
        for j in range(i+1,n):
            suma += macierzA[i][j] * wektorNiewiadomych[j]
        wektorNiewiadomych[i] = (wektorPrawy[i] - suma) / macierzA[i][i]

    return wektorNiewiadomych

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

    #a = np.linalg.solve(M,N)
    a = eliminacjaGaussa(M, N)
    return a

def wielomian_aproksymujacy(x,a,m):
    w = 0
    for i in range(m+1):
        w += a[i] * x ** i
    return w


a1 = wspolczynniki(2,X,Y)
a2 = wspolczynniki(1,X,Y)

xx = np.linspace(0.0,4.0,100)
yy = np.zeros([100])
yyy = np.zeros([100])

for i in range(100):
    yy[i] = wielomian_aproksymujacy(xx[i],a1, 2)

for i in range(100):
    yyy[i] = wielomian_aproksymujacy(xx[i],a2, 1)


plt.plot(xx,yy, label = 'aproksymacja st 2')
plt.plot(xx,yyy, label = 'aproksymacja st 1')
plt.scatter(X,Y, color = 'red', label = 'punkty')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Aproksymacja dla y = 0')
plt.legend()
plt.show()


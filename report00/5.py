import numpy as np
import matplotlib.pyplot as plt

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
            N[i] += Y[k] * X[k] ** i

    #a = np.linalg.solve(M,N)
    a = eliminacjaGaussa(M,N)
    return a

def wielomian_aproksymujacy(x,a,m):
    w = 0
    for i in range(m+1):
        w += a[i] * x ** i
    return w

data = np.loadtxt('siatka4.txt')
X = data[:, 0]
Y = data[:, 2]

a = wspolczynniki(2,X,Y)
a1 = wspolczynniki(1,X,Y)
print(a)
print(a1)

xx = np.linspace(min(X),max(X),10)
yy = np.zeros([10])
yyy = np.zeros([10])


for i in range(10):
    yy[i] = wielomian_aproksymujacy(xx[i],a, 2)
#roznice sa tak male ze nie widac ich na wykresie
print(yy)
for i in range(10):
    yyy[i] = wielomian_aproksymujacy(xx[i],a1, 1)
print(yyy)
plt.plot(xx,yy, label = 'aproksymacja st 2', color = 'black')
plt.plot(xx,yyy, label = 'aproksymacja st 1', color = 'lightgreen')
plt.title("Aproksymacja kwadratowa i liniowa dla y = 0.5 ")
plt.scatter(X,Y, color = 'red', label = 'punkty')
plt.xlabel('x')
plt.ylabel('F(x,y)')
plt.legend()
plt.show()

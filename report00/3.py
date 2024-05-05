import numpy as np
import matplotlib.pyplot as plt

file_path = "zad3.txt"
data = np.loadtxt(file_path)

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



def wsp(x,y):
    n = len(x)
    A = np.zeros([n,n])

    #tworzenie macierzy glownej
    for i in range(n):
        for j in range(n):
            A[i,j] = x[i]**j

    #rozwiazanie ukladu rownan
    #coef = np.linalg.solve(A,y)
    coef = eliminacjaGaussa(A,y) #WTF
    print(coef)

    return coef

def fun_bazowa(x,k):
    return x**k

def suma(a,x):
    summ = 0
    n = len(a)
    for i in range(n):
        summ += a[i] * fun_bazowa(x,i)
    return summ

X = data[:, 0]
Y = data[:, 2]

A = wsp(X,Y)

xx = np.linspace(min(X),max(X),100)
yy = np.zeros([100])

for i in range(100):
    yy[i] = suma(A,xx[i])

plt.plot(xx,yy)
plt.scatter(X, Y, color = 'green', label = 'wezly')
plt.title("Interpolacja wielomianowa dla y = 0")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
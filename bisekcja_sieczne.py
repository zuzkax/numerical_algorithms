import numpy as np
import matplotlib.pyplot as plt

dok = 1e-5

a = 1.069
b = 4.277
liczba_iteracji = 0

def f(x):
    return 8.51*x**2 + 0.44*x - 39.86

def bisekcja(a,b,poprzednik,dok, liczba_iteracji):
    x = (a+b)/2 

    if abs(x-poprzednik) < dok:
        return x, liczba_iteracji
    
    liczba_iteracji += 1

    if f(a) * f(x) < 0:
        plt.axvline(x,color = 'green', linewidth = 1)
        plt.text(x,f(x), liczba_iteracji,ha='center')
        return bisekcja(a,x,x,dok,liczba_iteracji)
    
    else:
        plt.axvline(x,color = 'green', linewidth = 1)
        plt.text(x,f(x), liczba_iteracji,ha='center')
        return bisekcja(x,b,x,dok,liczba_iteracji)
    
    

def roz_dok(a,b,c):
    delta = b**2 - 4*a*c
    if delta >= 0:
        x1 = (-b - np.sqrt(delta))/(2*a)
        x2 = (-b + np.sqrt(delta))/(2*a)

    else:
        print('brak miejsc zerowych')
    
    return x1, x2


def sieczne(a,b,x0,dok):
    liczba_iteracji = 0
    blad = 1000
    x = 0
    poprzednik = b
    while(blad>dok):
        x = poprzednik - ((f(poprzednik)*(x0-poprzednik))/(f(x0)-f(poprzednik)))
        blad = abs(x - poprzednik)
        poprzednik = x
        liczba_iteracji += 1
        plt.axvline(x,color = 'green', linewidth = 1)
        plt.text(x,f(x), liczba_iteracji,ha='center')

    xx = np.linspace(a,b,100)
    yy = np.zeros([100])
    for i in range(100):
        yy[i] = f(xx[i])
    plt.plot(xx,yy)
    plt.show()

    return x, liczba_iteracji

#print(f"metoda bisekcji: {bisekcja(a,b,1000,dok,liczba_iteracji)}")
a = 1.069
b = 4.277
#print(f"metoda siecznych: {sieczne(a,b,a,dok)}")
#print(f"delta:  {roz_dok(8.51,0.44,-39.86)}")

sieczne(a,b,a,dok)

xx = np.linspace(a,b,100)
yy = np.zeros([100])
for i in range(100):
    yy[i] = f(xx[i])

bisekcja(a,b,1000,dok,liczba_iteracji)
plt.plot(xx,yy)
plt.show()





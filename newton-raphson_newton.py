import numpy as np
def f(x):
    return 14.54 * x**2 - 8.29 * x - 49.05

def pochodna(x):
    return 2* 14.54 * x - 8.29

def newton(a,b,dok):
    x = a
    i = 0
    while(abs(f(x)) > dok):
        x = x - f(x) / pochodna(x)
        i += 1
    return x, i

def roz_dok(a,b,c):
    delta = b ** 2 - 4 * a * c
    if delta > 0:
        x1 = (-b - np.sqrt(delta)) / (2 * a)
        x2 = (-b + np.sqrt(delta)) / (2 * a)

        print(f"miejsca zerowe: x1 = {x1}, x2 = {x2}")
    elif delta == 0:
        print("jedno miejsce zerowe:", -b / (2 * a))
    else:
        print("brak miejsc zerowych")

def new_rap(a,b,dok):
    i = 0
    blad = 1000
    x = 0
    poprzednik = a
    while(blad>dok):
        x = poprzednik - (f(poprzednik)/pochodna(a))
        blad = abs(x - poprzednik)
        poprzednik = x
        i += 1
    return x, i

a = 14.54
b = -8.29
c = -49.05

p1 = 5
k1 = 10
p2 = -1
k2 = 2
dok = 1e-5
print(roz_dok(a,b,c))
print("metoda Newtona")
print(newton(p1,k1,dok))
print(newton(p2,k2,dok))
print("metoda Newtona-Raphsona")
print(new_rap(p1,k1,dok))
print(new_rap(p2,k2,dok))



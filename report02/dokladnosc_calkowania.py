import numpy as np
import matplotlib.pyplot as plt
from math import exp, pi


def f(x):
    return np.exp(x) * np.cos(x)

def trapezy(a, b, n):
    h = (b - a) / n
    wynik = (f(a) + f(b)) * (h / 2)
    suma = 0
    for i in range(1, n):
        suma += f(a + i * h)
    return (suma * h) + wynik


def simpson(a, b, n):
    h = (b - a) / n
    wynik = f(a) + f(b)
    suma1 = 0
    suma2 = 0
    for i in range(1, int(n / 2) + 1):
        suma1 += f(a + ((2 * i) - 1) * h)
    for j in range(1, int(n / 2)):
        suma2 += f(a + (2 * j) * h)
    return h / 3 * (wynik + 4 * suma1 + 2 * suma2)

a = 0
b = pi/2
n_war = [10, 100, 1000, 10000]
wyniki = []

I_dok = (exp(pi / 2) / 2) - (1 / 2)
print(I_dok)

for n in n_war:
    h = (b - a) / n
    I_t = trapezy(a, b, n)
    I_s = simpson(a, b, n)
    E_t = abs(I_dok - I_t)
    E_s = abs(I_dok - I_s)
    wyniki.append((n, h, I_t, I_s, E_t, E_s))

print(wyniki)

#wykres
h_wyniki = []
E_t_wyniki = []
E_s_wyniki = []
for i in range(4):
    h_wyniki.append(wyniki[i][1])
    E_t_wyniki.append(wyniki[i][4])
    E_s_wyniki.append(wyniki[i][5])

plt.loglog(h_wyniki, E_t_wyniki, 'o-', label='Błąd metody trapezów')
plt.loglog(h_wyniki, E_s_wyniki, 's-', label='Błąd metody Simpsona')
plt.xlabel('Krok całkowania (h)')
plt.ylabel('Błąd')
plt.legend()
plt.title('Zależność błędu od kroku całkowania')
plt.show()


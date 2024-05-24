import numpy as np 
import matplotlib.pyplot as plt

def f(x):
    return 11.22 * x**4 + 5.28 *x**3 + 16.33 *x**2 +1.83 *x + 5.1

def F(x):
    return 1/5*11.22*x**5 + 1/4 * 5.28 *x**4 + 1/3 * 16.33 *x**3 + 1/2 * 1.83 * x**2 + 5.1*x

def prostokaty(a,b,n):# n liczba przedzialow
    h = (b-a)/n # krok
    suma = 0
    for i in range(n):
        suma += f(a+i*h)
    return suma * h
    
def trapezy(a,b,n):
    h = (b-a)/n
    wynik = (f(a) + f(b))*(h/2)
    suma = 0
    for i in range(1,n):
        suma += f(a+i*h)
    return (suma * h) + wynik


def simpson(a,b,n):#?????
    h = (b-a)/n
    wynik = f(a)+ f(b)
    suma1 = 0
    suma2 = 0
    for i in range(1,int(n/2) + 1):
        suma1 += f(a+((2*i)-1)*h)
    for j in range(1,int(n/2)):
        suma2 += f(a+(2*j)*h)
    return h/3*(wynik + 4*suma1 + 2*suma2)

def analityczna(a,b):
    return F(b) - F(a)



print("analitycznie: ")
print(analityczna(-2,3))
print("metoda kwadratow:")
print(prostokaty(-2,3,1000))
print("metoda trapezow: ")
print(trapezy(-2,3,1000))
print("metoda simpsona: ")
print(simpson(-2,3,1000))

b = 0.001 
n_pocz = 10

#dla prostokatow
print("dla prostokatow")
b_wz = abs( (prostokaty(-2,3,n_pocz) - analityczna(-2,3))/analityczna(-2,3))

while b_wz >= b:
    n_pocz += 1
    b_wz = abs((prostokaty(-2,3,n_pocz) - analityczna(-2,3))/analityczna(-2,3)) 
print(f"n = {n_pocz}")

#dla trapezow
print("dla trapezow")
n_pocz = 10
b_wz = abs( (trapezy(-2,3,n_pocz) - analityczna(-2,3))/analityczna(-2,3))

while b_wz >= b:
    n_pocz += 1
    b_wz = abs((trapezy(-2,3,n_pocz) - analityczna(-2,3))/analityczna(-2,3)) 
print(f"n = {n_pocz}")

#dla simpsona
n_pocz = 10
print("dla simpsona")
b_wz = abs( (simpson(-2,3,n_pocz) - analityczna(-2,3))/analityczna(-2,3))

while b_wz >= b:
    n_pocz += 1
    b_wz = abs((simpson(-2,3,n_pocz) - analityczna(-2,3))/analityczna(-2,3)) 
print(f"n = {n_pocz}")

x = np.linspace(-2,3,100)
y = f(x)
plt.figure(figsize=(10,6))
plt.plot(x,y,'b',linewidth = 2, label = "f(x)")
h = (5)/100
for i in range(100):
    x0 = -2+i*h
    x1 =x0 +h
    y0 = f(x0)
    plt.fill_between([x0,x1],0,y0,color='pink', alpha = 0.3)
plt.xlabel('x')
plt.ylabel('y')
plt.title("metoda prostokatow")
plt.legend()
plt.show()

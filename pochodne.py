import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 5.31*x**3 - 4.55 *x ** 2 + 11.92 * x + 19.79


def pochodna(y,h):
    p = np.zeros([len(y)])

    p[0] = (y[1] - y[0])/h

    for i in range(1,len(y)-1):
        p[i] = (y[i+1]- y[i-1])/(2*h)

    p[len(y)-1] = (y[len(y)-1] - y[len(y)-2])/h

    return p


def pochodna2(y,h):
    p2 = np.zeros([len(y)])

    p2[0] = (y[2] - 2*y[1] + y[0])/h**2

    for i in range(1,len(y)-1):
        p2[i] = (y[i+1]- 2*y[i] + y[i-1])/h**2

    p2[len(y)-1] = (y[len(y)-1] - 2* y[len(y)-2] + y[len(y)-3])/h**2

    return p2


def analitycznie(x):
    return 5.31*3*x**2 - 4.55 *2*x + 11.92

x = np.linspace(-3, 3, 100)
y = np.zeros([100])
a = np.zeros([100])
for i in range(100):
    y[i] = f(x[i])

for i in range(100):
    a[i] = analitycznie(x[i])

h = x[1] - x[0]

p = pochodna(y,h)
p2 = pochodna2(y,h)

 
plt.plot(x,p,label = "pierwsza pochodna")
plt.plot(x,p2,label = "druga pochodna")
plt.plot(x,a, label = "pochodna 1 ana.")
#plt.plot(x,y,label = "f(x)")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()




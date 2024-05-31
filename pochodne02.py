import numpy as np
import matplotlib.pyplot as plt

def analitycznie(x):
    return np.cos(x)


def f(x):
    return np.sin(x)

def pochodna(x,t,eprz,ecent,h):
    
    for i in range(40):
        t[i] = (f(x + h[i]) - f(x))/(h[i])
        eprz[i] = abs(t[i] - analitycznie(x))
        ecent[i] = abs((f(x+h[i])-f(x-h[i]))/(2*h[i]) - analitycznie(x))
        h[i+1] = h[i]/2

    return t,eprz,ecent


t = np.zeros([40])
eprz = np.zeros([40])
ecent = np.zeros([40])
h = np.zeros([41])
h[0] = 0.1


t, eprz, ecent = pochodna(6,t,eprz,ecent,h)

print(h)

plt.loglog(h[0:40], t, label = "przyblizenie w x0 = 6")
plt.loglog(h[0:40], eprz, label = "blad przedni")
plt.loglog(h[0:40], ecent, label = "blad cent")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

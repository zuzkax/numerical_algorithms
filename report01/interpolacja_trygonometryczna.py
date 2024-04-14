import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x-1)**2 #funkcja nie jest okresowa

def wsp(x,y,n):
    a = np.zeros([n+1])
    b = np.zeros([n])

    for i in range(2*n+1):
        a[0] += 1/np.sqrt(2)*y[i]

    for k in range(1, n+1):
        for i in range(2*n+1):
            a[k] += np.cos(k*x[i])*y[i]
            b[k-1] += np.sin(k*x[i])*y[i]

    a = a*2/(2*n+1)
    b = b*2/(2*n+1)
    return a,b

def suma(a,b,n,x):
    sum = 0
    for i in range(1,n+1):
        sum += a[i]*np.cos(i*x) + b[i-1]*np.sin(i*x)
    sum += a[0]/(np.sqrt(2))
    return sum

l = input("podaj liczbe punktow")

n = int(l)//2
print('liczba n:' , n)

m = 2*n +1
X = np.zeros([m])
Y = np.zeros([m])

for i in range(m):
    X[i] = (2*i*np.pi)/(2*n+1)
    Y[i] = f(X[i])

a,b = wsp(X,Y,n)

xx = np.linspace(0, 2*np.pi, 100)
yy = np.zeros([100])
for i in range(100):
    yy[i] = suma(a,b,n,xx[i])

plt.plot(xx,yy)
#plt.plot(X,Y,'o')
plt.scatter(X, Y, color= 'red', label = 'wezly')
plt.title("Wielomian trygonometryczny")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
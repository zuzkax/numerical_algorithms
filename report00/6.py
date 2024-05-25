import numpy as np

wsp_ap1 = [550.06827666, -82.80063589]
#kontrolna = [11.22,5.28,16.33,1.83,5.1]
#kontrolna_reverse = kontrolna[::-1]
wsp_ap2 = [5.48913574e+02, -8.10241696e+01, -4.44116578e-01]
#wsp_wiel =
#wsp_spl =



def f(x,wsp):
    output = 0
    for i in range(len(wsp)):
        output += wsp[i] * x**i
    return output

def trapezy(a,b,n,w):
    h = (b-a)/n
    wynik = (f(a,w) + f(b,w))*(h/2)
    suma = 0
    for i in range(1,n):
        suma += f(a+i*h, w)
    return (suma * h) + wynik

def simpson(a,b,n,w):
    h = (b-a)/n
    wynik = f(a,w)+ f(b,w)
    suma1 = 0
    suma2 = 0
    for i in range(1,int(n/2) + 1):
        suma1 += f(a+((2*i)-1)*h,w)
    for j in range(1,int(n/2)):
        suma2 += f(a+(2*j)*h,w)
    return h/3*(wynik + 4*suma1 + 2*suma2)

data = np.loadtxt('siatka5.txt')
X = data[:, 0]

print("metoda trapezow dla wspolczynnikow aproksymacji 1 stopnia: ")
print(trapezy(min(X),max(X),1000,wsp_ap1[::-1]))
print("metoda simpsona dla wspolczynnikow aproksymacji 1 stopnia: ")
print(simpson(min(X),max(X),1000,wsp_ap1[::-1]))
#print(trapezy(-2,3,1000,kontrolna_reverse))
print("metoda trapezow dla wspolczynnikow aproksymacji 2 stopnia: ")
print(trapezy(min(X),max(X),1000,wsp_ap2[::-1]))
print("metoda simpsona dla wspolczynnikow aproksymacji 2 stopnia: ")
print(simpson(min(X),max(X),1000,wsp_ap2[::-1]))


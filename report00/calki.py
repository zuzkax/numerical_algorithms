import numpy as np

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

def calka_s_trapez(x, y):
    c = 0
    for i in range(1, len(x)):
        c += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2
    return c

def calka_s_simpson(x, y):
    c = 0
    for i in range(1, len(x)-1, 2):
        h = x[i] - x[i-1]
        c += (h / 3) * (y[i-1] + 4 * y[i] + y[i+1])
    return c

def calka_l_simpson(x, y):
    integral = 0
    if len(x) % 2 == 0:
        x = np.append(x, x[-1] + (x[-1] - x[-2]))
        y = np.append(y, 0)
    for i in range(1, len(x)-1, 2):
        h = x[i] - x[i-1]
        integral += (h / 3) * (y[i-1] + 4 * y[i] + y[i+1])
    return integral

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
def BS(xx, x, h):
    if x-2*h <= xx <= x -h:
        return 1 / h**3 * (xx -(x - 2 *h))**3
    elif x-h <= xx <= x:
        return 1 / h**3 * (h**3 + 3 *(h**2) * (xx-(x-h)) + 3*h*(xx-(x-h))**2 - 3 * (xx-(x-h))**3)
    elif x <= xx <= x + h:
        return 1 / h**3 * (h**3 + 3 * h**2 * ((x+h)-xx) + 3*h*((x+h)-xx)**2 -3 * ((x+h)-xx)**3)
    elif x +h <= xx <= x+2 * h:
        return 1/h **3 * (((x + 2 *h)-xx)**3)
    else:
        return 0
def splajny(X,Y):
    x = np.zeros(len(X)+2)
    y = np.zeros(len(Y)+2)
    x[1:len(X)+1] = X
    y[1:len(Y) + 1] = Y
    y[0] = 1.0
    y[len(Y) + 1] = -1.0
    h = X[2] - X[1]
    x[0] = x[0] -h
    x[len(X) + 1] = x[len(X) - 1] + h
    A = np.zeros((len(X)+ 2, len(X)+2))
    A[0,0] = -3.0/h
    A[0,2] = 3.0/h
    A[len(X) + 1, len(X) - 1] = -3.0 /h
    A[len(X) + 1, len(X) +1] = 3.0 / h
    for i in range(1,len(X)+1):
        A[i,i+1] = 1.0
        A[i,i] = 4.0
        A[i,i-1] = 1.0

    M = eliminacjaGaussa(A,y)
    xx = np.linspace(min(X),max(X),1000)
    yy = np.zeros_like(xx)

    for i in range(len(xx)):
        summ = 0
        for j in range(len(M)):
            summ += M[j] * BS(xx[i],x[j],h)
        yy[i] = summ
    return xx, yy

def LagrangeVal(macierzWsp, macierzX, x):
    n = macierzWsp.shape[0]
    y = 0
    for i in range(n):
        m = 1
        for j in range(n):
            if j != i:
                m = m * (x - macierzX[j])
        y = y + m * macierzWsp[i]
    return y
def LagrangeCoeff(X, Y):
    n = np.size(X)
    a = np.zeros(n)
    for i in range(n):
        m = 1
        for j in range(n):
            if j != i:
                m = m * (X[i] - X[j])
        a[i] = Y[i] / m
    return a
def interpolacja_przedzialami(X, Y, segment_size=5):
    n = len(X)
    xx = []
    yy = []
    wsp = []

    i = 0
    while i < n:
        end = i + segment_size
        if end > n:
            end = n
        X_segment = X[i:end]
        Y_segment = Y[i:end]
        a = LagrangeCoeff(X_segment, Y_segment)
        wsp.append(a)
        print(f"Współczynniki dla przedziału {i} do {end}: {a}")
        xx_segment = np.linspace(X_segment[0], X_segment[-1], 100)
        yy_segment = [LagrangeVal(macierzWsp=a, macierzX=X_segment, x=xi) for xi in xx_segment]

        xx.extend(xx_segment)
        yy.extend(yy_segment)

        if end == n:
            break
        i = end - 1

    return np.array(xx), np.array(yy)

data = np.loadtxt('siatka5.txt')
X = data[:, 0]
Y = data[:, 2]


wsp_ap1 = [550.06827666, -82.80063589]
wsp_ap2 = [5.48913574e+02, -8.10241696e+01, -4.44116578e-01]

xxS, yyS = splajny(X,Y)
xxL, yyL = interpolacja_przedzialami(X,Y)

print("metoda trapezow dla  aproksymacji 1 stopnia: ")
print(trapezy(min(X),max(X),1000,wsp_ap1[::-1]))
print("metoda simpsona dla  aproksymacji 1 stopnia: ")
print(simpson(min(X),max(X),1000,wsp_ap1[::-1]))
print("metoda trapezow dla  aproksymacji 2 stopnia: ")
print(trapezy(min(X),max(X),1000,wsp_ap2[::-1]))
print("metoda simpsona dla  aproksymacji 2 stopnia: ")
print(simpson(min(X),max(X),1000,wsp_ap2[::-1]))
print("===========================================================")
print(f"metoda trapezow dla interpolacji Lagrangea: {calka_s_trapez(xxL,yyL)} ")
print(f"metoda simpsona dla interpolacji Lagrangea: {calka_l_simpson(xxL,yyL)} ")
print("===========================================================")
print(f"metoda trapezow dla splajnu: {calka_s_trapez(xxS,yyS)}")
print(f"metoda simpsona dla splajnu: {calka_s_simpson(xxS,yyS)}")

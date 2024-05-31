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


data = np.loadtxt('siatka5.txt')
X = data[:, 0]
Y = data[:, 2]

calki_lag_trapezy = []
calki_lag_sim = []

wsp_ap1 = [550.06827666, -82.80063589]
wsp_ap2 = [5.48913574e+02, -8.10241696e+01, -4.44116578e-01]
wsp_lag= [[  198498.75    ,    -905245.     ,     1681315.     ,    -1252991.66666667,
   278269.58333333], [  278269.58333333, -1288688.33333333 , 2245372.5  ,      -1065738.33333333,
   235390.83333333],[  235390.83333333, -1325855. ,         1291685.       ,   -299183.33333333,
   205062.5       ],[ 205062.5 ,       -734653.33333333 ,-252585.    ,     -266388.33333333,
  214363.75      ], [  214363.75    ,      66373.33333333 ,  -74652.  ,       -1165156.66666667,
   159748.33333333],[  159748.33333333  ,  18780.33333333  ,2050510.00000001 ,-1573661.66666667,
    66810.41666667], [   66810.41666667, -1153803.33333333  ,3273000.  ,        -633210.,
   116027.08333333], [  116027.08333333, -2073483.33333333 , 1446674.99999999   ,395671.66666667,
   315092.5       ], [  315092.5   ,     -1250055.   ,      -1340267.5    ,     -139062.33333333,
   375185.        ], [  375185.        ,   607755.   ,     -1132530.    ,     -1635320.,
   115332.08333333]]

calka_lag_trapezy = 0
calka_lag_sim = 0

for i in range(len(wsp_lag)):
    calki_lag_trapezy.append(trapezy(min(X), max(X),1000,wsp_lag[i]))
    calki_lag_sim.append(simpson(min(X), max(X),1000,wsp_lag[i]))
    calka_lag_sim += simpson(min(X), max(X),1000,wsp_lag[i])
    calka_lag_trapezy += trapezy(min(X), max(X),1000,wsp_lag[i])

#print(calki_lag_trapezy)
#print(calki_lag_sim)

xxS, yyS = splajny(X,Y)


print("metoda trapezow dla  aproksymacji 1 stopnia: ")
print(trapezy(min(X),max(X),1000,wsp_ap1[::-1]))
print("metoda simpsona dla  aproksymacji 1 stopnia: ")
print(simpson(min(X),max(X),1000,wsp_ap1[::-1]))
print("metoda trapezow dla  aproksymacji 2 stopnia: ")
print(trapezy(min(X),max(X),1000,wsp_ap2[::-1]))
print("metoda simpsona dla  aproksymacji 2 stopnia: ")
print(simpson(min(X),max(X),1000,wsp_ap2[::-1]))
print("===========================================================")
print(f"metoda trapezow dla interpolacji Lagrangea: {calka_lag_trapezy} ")
print(f"metoda simpsona dla interpolacji Lagrangea: {calka_lag_sim} ")
print("===========================================================")
print(f"metoda trapezow dla splajnu: {calka_s_trapez(xxS,yyS)}")
print(f"metoda simpsona dla splajnu: {calka_s_simpson(xxS,yyS)}")

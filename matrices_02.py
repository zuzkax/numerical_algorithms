import numpy as np


macierzA = np.array([[7.0,-2.0,8.0,6.0,4.0,-9.0],[-7.0,-3.0,8.0,-9.0,7.0,-6.0],[9.0,-3.0,-1.0,9.0,9.0,6.0],[3.0,-9.0,7.0,8.0,2.0,-5.0],[-1.0,-7.0,-8.0,0.0,-8.0,-10.0],[1.0,6.0,5.0,2.0,3.0,-8.0]])
wektorPrawy = np.array([9.0,3.0,4.0,-4.0,6.0,9.0])

macierzAcp = macierzA.copy()
wektorPrawycp = wektorPrawy.copy()

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
print(wektorPrawy)

#print("wynik z funkcji wbudowanej" + "\n")
print("wynik z funkcji wbudowanej", np.linalg.solve(macierzAcp, wektorPrawycp))

#print("wynik z funkcji napisanej" + "\n")
print("wynik z funkcji napisanej", eliminacjaGaussa(macierzA=macierzA, wektorPrawy=wektorPrawy))


wektorDolny = np.array([0.0,-1.0,7.0,-1.0,6.0,7.0])
wektorGlowny = np.array([7.0,-3.0,4.0,7.0,-7.0,-6.0])
wektorGorny = np.array([0.0,-2.0,-7.0,3.0,9.0,0.0])
wektorPrawychStron =  np.array([2.0,2.0,-2.0,-4.0,7.0,9.0])
def metodaThomasa(a,b,c,d) :
    n = 6
    wektorNiewiadomych = np.zeros([n])

    bet = np.zeros([n])
    gam = np.zeros([n])
    
    bet[0] = -c[0]/b[0]
    gam[0] = d[0]/b[0]

    for i in range(1,n):
        bet[i] = -c[i]/(a[i]* bet[i-1] + b[i])
        gam[i] = (d[i] - a[i] * gam[i-1])/(a[i]* bet[i-1] + b[i])

    wektorNiewiadomych[n-1] = gam[n-1]
    for i in range(n-2,-1,-1):
        wektorNiewiadomych[i] = gam[i] + bet[i] * wektorNiewiadomych[i+1]

    return wektorNiewiadomych


print(metodaThomasa(wektorDolny,wektorGlowny,wektorGorny, wektorPrawychStron))

            

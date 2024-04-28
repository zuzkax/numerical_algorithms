import numpy as np

macierzA = np.random.rand(3,2) #3 kolunny 2 wiersze
macierzB = np.random.rand(2,3) #2 kolumny 3 wiersze
macierzD = np.array([[9,0.6,7,12.7],[3.33,1,-8,19],[1.97,6,5,-4],[1,-1,2.5,0.9]])

def mnozenie_macierzy(macierzA, macierzB):

    mA = macierzA.shape[0] #3
    nA = macierzA.shape[1] #2
    mB = macierzB.shape[1] #3
    nB = macierzB.shape[0] #2


    if nA == nB:
        macierzC = np.zeros([mA,mB])

        for j in range(mA):
            for i in range(mB):
                for s in range(nA):
                    macierzC[j][i] = macierzC[j][i] + macierzA[j][s] * macierzB[s][i]

        return macierzC
    else:
        #print(nA)
        #print(nB)
        return "nie mozna wykonac dzielenia"
        
        
    
def wyznacznik_macierzy(macierzD):

    N = macierzD.shape[0]

    d = macierzD[0][0]

    for s in range(N-1):
        for i in range(s+1,N):
            for j in range(s+1,N):
                macierzD[i][j] = macierzD[i][j] - (macierzD[i][s]/macierzD[s][s]) * macierzD[s][j]
        d = d * macierzD[s+1][s+1]

    return d


def macierz_odwrotna(macierzD):
    N = macierzD.shape[0]
    macierzB = np.zeros([N,2*N])
    E = np.zeros([N,N])

    for j in range(N):
        for i in range(N):
            macierzB[j][i] = macierzD[j][i]

    for j in range(N):
        for i in range(N+1,2*N):
            macierzB[j][i] = 0

    for j in range(N):
        macierzB[j][j+N] = 1

    for s in range(N):
        c = macierzB[s][s]
        macierzB[s][s] = macierzB[s][s] -1 

        for j in range(s+1, 2*N):
            d = macierzB[s][j]/c

            for i in range(N):
                macierzB[i][j] = macierzB[i][j] - (d*macierzB[i][s])

    E[0:N,0:N] = macierzB[0:N,N:2*N]
    return E
 



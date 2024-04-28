import numpy as np

A = np.array([[14.89, 4.07, -2.09, 2.98, 1.0],[0.58, 10.72, -4.19, -2.06, 2.53],[2.89, 2.81, 11.41, -0.43, 3.93],[-1.14, -2.26, -2.56, 13.73, 4.07],[4.09, 2.73, -3.14,- 0.41, 11.78]])
B = np.array([-6.38,1.21,-7.22,-4.17,2.86])
A_cp = A.copy()
B_cp = B.copy()

l = B.shape[0]
x0 = np.zeros([l])
dokladnosc = 1e-10

def iteracja_prosta(A,B,x0, dokladnosc):
    x0 = x0.copy()
    A=A.copy()
    liczba_iteracji = 0
    l = x0.shape[0]
    D = np.zeros([l,l])
    R = A

    for i in range(l):
        D[i][i] = A[i][i]
        R[i][i] = 0
    
    M = - np.dot(np.linalg.inv(D),R)
    P = np.dot(np.linalg.inv(D),B)

   
    spr = 1000
    while(spr > dokladnosc):
        X_P = x0.copy()
        x0 = np.dot(M,x0) + P
        spr = np.sum(abs(X_P-x0))/5
        liczba_iteracji +=1

    return x0, liczba_iteracji

def metoda_GS(A, B,x0,dokladnosc):
    l = x0.shape[0]
    spr = 1000
    liczba_iteracji = 0
    while(spr > dokladnosc):
        liczba_iteracji += 1
        x0_p = x0.copy()

        for i in range(l):
            sum1 = 0
            sum2 = 0
            for j in range(i):
                sum1 += A[i][j] * x0[j]
            for j in range(i+1,l):
                sum2 += A[i][j] * x0_p[j]

            x0[i] = 1/A[i][i]*(B[i]-sum1-sum2)
        
        spr = np.sum(abs(x0_p-x0))/5
    return x0, liczba_iteracji

print("wynik z iteracji prostej:")
print(iteracja_prosta(A,B,x0,dokladnosc))
print("wynik dokładny")
print(np.linalg.solve(A_cp,B_cp))
print("wynik z metody Gaussa-Seidela")
print(metoda_GS(A,B,x0,dokladnosc))
print("metoda Gaussa-Seidela szybciej uzyskuje wynik")



    
import numpy as np
import matplotlib.pyplot as plt
from plots import init, getC, getHist

def PCA(x,m):
    """PCA come in part2.ipynb: ritorna matrice di proiezione e dati proiettati."""
    C=getC(x)
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = np.dot(P.T, x)
    return P,DP

if __name__ =="__main__":
    D,L,labels=init()
    U,DP=PCA(D,6)
    getHist(DP,L,labels)
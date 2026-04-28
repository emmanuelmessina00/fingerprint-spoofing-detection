import numpy as np
import matplotlib.pyplot as plt
import scipy

from plots import init, getMu, vCol, vRow,getC, getHist

def compute_Sb_Sw(D, L):
    Sw=0
    Sb=0
    N=D.shape[1]
    mu=vCol(getMu(D))
    for c in np.unique(L):
        mask=L==c
        D0=D[:,mask]
        nc=D0.shape[1]
        mc=vCol(getMu(D0))
        Sb+=nc*((mc-mu)@(mc-mu).T)
        DC=D0-mc
        Sw+=(DC @ DC.T)
    return Sb/N,Sw/N
        
def LDA(D,L,classes,m):
    Sb,Sw=compute_Sb_Sw(D,L)
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    DP=W.T @ D
    return W,DP

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)
# DTR and LTR are model training data and labels
# DVAL and LVAL are validation data and labels

if __name__ =="__main__":
    
    D,L,labels=init()
    W,Dlda=LDA(D,L,labels,1)
    getHist(Dlda,L,labels)
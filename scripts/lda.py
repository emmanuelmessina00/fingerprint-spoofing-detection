import numpy as np
import matplotlib.pyplot as plt
import scipy
from scripts.utils import *

def train_LDA(X,L,m):
    Sb,Sw=compute_Sb_Sw(X,L)
    _,U=scipy.linalg.eigh(Sb,Sw)
    W=U[:,::-1][:,:m]
    return W
def apply_LDA(X,W):
    return W.T @ X

if __name__ =="__main__":
    
    D,L,labels=init()
    W,Dlda=LDA(D,L,labels,1)
    getHist(Dlda,L,labels)
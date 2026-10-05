
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import scipy

from scripts.utils import *
from scripts.logpdf_GAU_ND import *


def get_params_MVG(DTR,LTR):

    params=[]

    for i in np.unique(LTR):
        D0=DTR[:,LTR==i]
        mu=getMu(D0)
        C=getC(D0)
        params.append((mu,C))

    return params

def get_params_Naive_Bayes(DTR,LTR):
    params=[]

    for i in np.unique(LTR):
        D0=DTR[:,LTR==i]
        mu=getMu(D0)
        C=getC(D0) * np.eye(DTR.shape[0])
        params.append((mu,C))

    return params

def get_params_Tied_Covariance(DTR,LTR):
    params=[]
    for i in np.unique(LTR):
        D0 = DTR[:,LTR==i]
        mu = getMu(D0)
        params.append(mu)
    
    _,Sw=compute_Sb_Sw(DTR,LTR)
    params.append(Sw)
    return params
def getS(DVAL,params):
    res=[]

    for i in range(len(params)):
        mu=np.array(params[i][0]).reshape(DVAL.shape[0],1)
        res.append(logpdf_GAU_ND(DVAL,mu,params[i][1]))
    
    logS=np.stack(res)
    return logS

def getS_Tied_Covariance(DVAL,params):
    res=[]

    for i in range(len(params)-1):
        mu=np.array(params[i]).reshape(DVAL.shape[0],1)
        res.append(logpdf_GAU_ND(DVAL,mu,params[len(params)-1]))
    
    logS=np.stack(res)
    S=np.exp(logS)
    return S
def MVG_Classifier(DTR,LTR,DVAL,LVAL):
    params=get_params_MVG(DTR,LTR)
    S_MVG=getS(DVAL,params)
    log_false=S_MVG[0]
    log_true=S_MVG[1]
    s=log_true-log_false
    predictions_MVG=np.where(s>=0,1,0)
    accuracy=np.where(predictions_MVG == LVAL,1,0).mean()
    err=1-accuracy
    print(f"The error rate for Binary Classifier based on MVG is: {err*100}% and the accuracy: {accuracy*100}%")

    return predictions_MVG
def Naive_Bayes_Classifier(DTR,LTR,DVAL,LVAL):
    params_Naive_Bayes=get_params_Naive_Bayes(DTR,LTR)

    S_NaiveBayes=getS(DVAL,params_Naive_Bayes)
    log_false=S_NaiveBayes[0]
    log_true=S_NaiveBayes[1]
    s=log_true-log_false
    predictions_Naive_Bayes=np.where(s>=0,1,0)
    accuracy=np.where(predictions_Naive_Bayes == LVAL,1,0).mean()
    err=1-accuracy

    print(f"The error rate for Binary Classifier based on Naive Bayes is: {err*100}% and the accuracy: {accuracy*100}%")
    return predictions_Naive_Bayes
def Tied_Covariance_Classifier(DTR,LTR,DVAL,LVAL):
    params_TCG=get_params_Tied_Covariance(DTR,LTR)
    S_TGC=getS_Tied_Covariance(DVAL,params_TCG)
    log_false=S_TGC[0]
    log_true=S_TGC[1]
    s=log_true-log_false
    predictions_TCG=np.where(s>=0,1,0)
    accuracy=np.where(predictions_TCG == LVAL,1,0).mean()
    err=1-accuracy

    print(f"The error rate for Binary Classifier based on Tied Covariance Matrix is: {err*100}% and the accuracy: {accuracy*100}%")
    return  predictions_TCG



def vCol(vet):
    return vet.reshape((vet.size, 1))

def compute_correlations(params, labels):
    """Compute and display correlation matrices for each class"""
    for i in range(len(params)):
        C = params[i][1]
        Corr = C / (vCol(C.diagonal()**0.5) * vRow(C.diagonal()**0.5))
        print(f"\n{'='*50}")
        print(f"Class {i} ({labels[i]})")
        print(f"{'='*50}")
        print(f"Correlation Matrix:\n{Corr}\n")

if __name__ == "__main__":
    pass
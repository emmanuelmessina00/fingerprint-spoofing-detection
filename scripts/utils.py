import numpy as np

def init_dataset():
    dataset = np.loadtxt('../trainData.txt', delimiter=',')
    D = dataset[:, :6].T
    L = dataset[:, 6]

    labels = {
        1: "True",
        0: "False"
    }

    return D, L, labels

def getMu(D):
    return D.sum(axis=1)/D.shape[1]

def vRow(vet):
    return vet.reshape((1, vet.size))
def vCol(vet):
     return vet.reshape((vet.size, 1))

def getC(D):
    mu=getMu(D)
    mu=vCol(mu)
    Dc=D-mu
    return (Dc @ Dc.T)/Dc.shape[1]

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

def get_confusion_mat(predictions, L):
    labels = np.unique(L)
    M = np.zeros((len(labels), len(labels)))

    predictions_int = np.array(predictions).astype(int)
    L_int = np.array(L).astype(int)

    for i in range(len(predictions_int)):
        M[predictions_int[i]][L_int[i]] += 1

    return M
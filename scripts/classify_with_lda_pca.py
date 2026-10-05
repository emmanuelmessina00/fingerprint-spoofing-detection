import numpy as np
import matplotlib.pyplot as plt
import scipy

from scripts.utils import *
from scripts.lda import *
from scripts.pca import *

def classify_with_LDA(DTR,LTR,DVAL,LVAL):
    # Calcola LDA SOLO sul training set
    W=train_LDA(DTR,LTR,1)
    DTR_lda = apply_LDA(DTR,W)

    # Controlla l'orientamento: la media di True (1) deve essere > media di False (0)
    mean_false = DTR_lda[0, LTR==0].mean()
    mean_true = DTR_lda[0, LTR==1].mean()

    # Se l'orientamento è sbagliato, inverti W
    if mean_true < mean_false:
        W = -W
        DTR_lda = W.T @ DTR
        mean_false = DTR_lda[0, LTR==0].mean()
        mean_true = DTR_lda[0, LTR==1].mean()

    # Applica la stessa trasformazione W ai dati di validazione
    DVAL_lda = W.T @ DVAL

    # Calcola threshold come media delle medie di classe
    threshold = (mean_false + mean_true) / 2.0

    # Predizioni
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 1
    PVAL[DVAL_lda[0] < threshold] = 0

    print('Threshold:', threshold)
    print('Mean False:', mean_false, 'Mean True:', mean_true)
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of errors:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))


def classify_with_PCA_and_LDA(D, L, m):
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # PCA sul training set
    P, mu = train_PCA(DTR, m)
    DTR_pca = apply_PCA(DTR, P, mu)
    DVAL_pca = apply_PCA(DVAL, P, mu)

    # LDA sul training set PCA-preprocessato
    Sb_tr, Sw_tr = compute_Sb_Sw(DTR_pca, LTR)
    s, U = scipy.linalg.eigh(Sb_tr, Sw_tr)
    W = U[:, ::-1][:, 0:1]

    # Controlla orientamento
    DTR_lda = W.T @ DTR_pca
    mean_false = DTR_lda[0, LTR == 0].mean()
    mean_true = DTR_lda[0, LTR == 1].mean()

    if mean_true < mean_false:
        W = -W
        DTR_lda = W.T @ DTR_pca
        mean_false = DTR_lda[0, LTR == 0].mean()
        mean_true = DTR_lda[0, LTR == 1].mean()

    # Applica LDA ai dati di validazione PCA-preprocessati
    DVAL_lda = W.T @ DVAL_pca

    # Threshold e classificazione
    threshold = (mean_false + mean_true) / 2.0
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 1
    PVAL[DVAL_lda[0] < threshold] = 0

    error_rate = (PVAL != LVAL).sum() / float(LVAL.size) * 100
    return error_rate

if __name__ =="__main__":
    pass
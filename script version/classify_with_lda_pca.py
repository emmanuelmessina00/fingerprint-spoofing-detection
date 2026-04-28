import numpy as np
import matplotlib.pyplot as plt
import scipy

from plots import init
from lda import LDA, compute_Sb_Sw,split_db_2to1
from pca import PCA

def classify_with_LDA(D,L):
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    unique_classes = np.unique(LTR)
    # Calcola LDA SOLO sul training set
    W,DTR_lda = LDA(DTR,LTR,unique_classes,1)

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
    P, DTR_pca = PCA(DTR, m)
    
    # Applica la stessa proiezione PCA ai dati di validazione
    DVAL_pca = P.T @ DVAL
    
    # LDA sul training set PCA-preprocessato
    unique_classes = np.unique(LTR)
    W, DTR_lda = LDA(DTR_pca, LTR, unique_classes, 1)
    
    # Controlla orientamento
    mean_false = DTR_lda[0, LTR==0].mean()
    mean_true = DTR_lda[0, LTR==1].mean()
    
    if mean_true < mean_false:
        W = -W
        DTR_lda = W.T @ DTR_pca
        mean_false = DTR_lda[0, LTR==0].mean()
        mean_true = DTR_lda[0, LTR==1].mean()
    
    # Applica LDA ai dati di validazione PCA-preprocessati
    DVAL_lda = W.T @ DVAL_pca  # FIX: usa DVAL_pca!
    
    # Threshold e classificazione
    threshold = (mean_false + mean_true) / 2.0
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 1
    PVAL[DVAL_lda[0] < threshold] = 0
    
    error_rate = (PVAL != LVAL).sum() / float(LVAL.size) * 100
    return error_rate

# Analizza le prestazioni in funzione di m



if __name__ =="__main__":
    D,L,labels=init()
    print("\nAnalisi LDA: Error rate\n")
    classify_with_LDA(D,L)
    print("\nAnalisi PCA + LDA: Error rate vs numero di dimensioni PCA\n")
    for m in range(1, 7):
        error_rate = classify_with_PCA_and_LDA(D, L, m)
        print(f"m = {m}: Error rate = {error_rate:.1f}%")
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import scipy
from plots import vRow,getC,getMu,init

def logpdf_GAU_ND(x, mu, C):
    M=x.shape[0]
    XC = x - mu
    sign, logdet = np.linalg.slogdet(C)
    invC = np.linalg.inv(C)
    quad = np.sum(XC * (invC @ XC), axis=0)
    return -0.5 * (M * np.log(2*np.pi) + logdet + quad)

def loglikelihood(XND, m_ML, C_ML):
    logpdf=logpdf_GAU_ND(XND,m_ML,C_ML)
    return np.sum(logpdf)

if __name__=="__main__":
    D,L,labels=init()

    n_features = D.shape[0]
    for i in range(n_features):
        plt.figure(figsize=(8, 5))

        for cnt in range(2):
            mask = L == cnt
            D0 = vRow(D[i, mask])
            mu = getMu(D0)
            C = getC(D0)

            # 1. Disegna l'istogramma
            plt.hist(D0.ravel(), bins=50, density=True, alpha=0.5, label=f'Class {labels[cnt]} (Hist)')

            # 2. Crea un asse X ordinato che va dal minimo al massimo dei tuoi dati per disegnare la curva
            XPlot = np.linspace(D0.min(), D0.max(), 1000)

            # 3. Calcola la log-densità su XPlot (ricorda di usare vRow) ed esponenziala
            pdf_values = np.exp(logpdf_GAU_ND(vRow(XPlot), mu, C))

            # 4. Disegna la curva Gaussiana
            plt.plot(XPlot.ravel(), pdf_values.ravel(), linewidth=2, label=f'Class {labels[cnt]} (Gaussian)')

        plt.title(f'Distribuzione Feature {i} con fit Gaussiano per classe')
        plt.xlabel(f'Feature {i}')
        plt.ylabel('Densità')
        plt.legend()
        plt.tight_layout()
        plt.show()
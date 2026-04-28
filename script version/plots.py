import numpy as np
import matplotlib.pyplot as plt
import os

def init():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = np.loadtxt(os.path.join(script_dir, 'trainData.txt'), delimiter=',')
    D=dataset[:,:6].T
    L=dataset[:,6]

    labels={
        1:"True",
        0:"False"
    }

    return D,L,labels

def getHist(D,L,classes):
    n_features = D.shape[0]  # numero effettivo di feature
    for i in range(n_features):
        plt.figure()
        plt.ylabel("density")
        plt.xlabel(f"Feature {i}")
        for cnt in range(2):
            mask= L==cnt
            # D[:, mask] estrae tutti i campioni di quella classe
            # [i, :] estrae solo i valori della i-esima feature per quei campioni
            feature_data = D[:, mask][i, :]
            plt.hist(feature_data, bins=30, density=True, alpha=0.5, label=f"{classes[cnt]}")
        
        plt.legend()  # Add legend after plotting all classes
        plt.title(f"Histogram of feature {i}")
        plt.show()  # Display the histogram

def getScatters(D,classes):
    n_features = D.shape[0]
    for i in range(n_features):
        for j in range(i + 1, n_features):
            plt.figure()
            plt.title(f"Scatter Plot: Feature {i} vs Feature {j}")
            plt.xlabel(f"Feature {i}")
            plt.ylabel(f"Feature {j}")
            
            for cnt in range(2):
                mask= L==cnt
                D0=D[:,mask]
                name=classes[cnt]
                plt.scatter(D0[i,:],D0[j,:],alpha=0.4, label=name)
                
            plt.legend()
            plt.show() 

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


if __name__ =="__main__":

    D,L,labels=init()
    
    labels={
        1:"True",
        0:"False"
    }

    getHist(D,L,labels)
    getScatters(D,labels)
    D0=D[:,L==0]
    D1=D[:,L==1]

    mu0=getMu(D0)
    mu1=getMu(D1)
    mu=getMu(D)

    C0=getC(D0)
    C1=getC(D1)
    C=getC(D)

    print(f"Le medie sono, media globale {mu}, media per i falsi {mu0} e media per i veri {mu1}")
    print(f"Le covarianze sono, covarianza globale {C}, covarianza per i falsi {C0} e covarianza per i veri {C1}")

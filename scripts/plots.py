import numpy as np
import matplotlib.pyplot as plt
import os
from scripts.dcf import *
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

def getScatters(D,classes, L):
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

def plot_bayes_error(models, LVAL):
    effPriorLogOdds = np.linspace(-4, 4, 21)
    plt.figure(figsize=(10, 7))

    for model_name, model_llr in models.items():
        dcf = []
        mindcf = []
        for p in effPriorLogOdds:
            pi = 1 / (1 + np.exp(-p))
            predictions = get_bayes_decision(model_llr, pi, 1, 1)
            dcf.append(DCF_norm(predictions, LVAL, pi, 1, 1))
            mindcf.append(compute_min_dcf(model_llr, LVAL, pi, 1, 1))

        line = plt.plot(effPriorLogOdds, dcf, label=f"DCF {model_name}")
        plt.plot(
            effPriorLogOdds,
            mindcf,
            label=f"min DCF {model_name}",
            color=line[0].get_color(),
            linestyle="--",
        )

    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
if __name__ =="__main__":
    pass
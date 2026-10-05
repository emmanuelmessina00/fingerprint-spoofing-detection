import numpy as np
import matplotlib.pyplot as plt
import os

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

if __name__ =="__main__":
    pass
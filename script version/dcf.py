from lda import split_db_2to1
from plots import init
from gaussianclass import (
    MVG_Classifier,
    getS,
    get_params_MVG,
    get_params_Naive_Bayes,
    get_params_Tied_Covariance,
    getS_Tied_Covariance,
)
import numpy as np
import matplotlib.pyplot as plt

def get_confusion_mat(predictions,L):
    labels=np.unique(L)
    M=np.zeros((len(labels),len(labels)))

    predictions_int = np.array(predictions).astype(int)
    L_int = np.array(L).astype(int)

    for i in range(len(predictions_int)):
        M[predictions_int[i]][L_int[i]]+=1
    
    return M

def DCF_u(predictions,L,prior,Cfn,Cfp):
    C=np.array([[0,Cfn],[Cfp,0]])
    M=get_confusion_mat(predictions,L)
    cols_sum=np.sum(M,axis=0)
    R=M/cols_sum 
    expected_cost_per_class=np.sum(R*C,axis=0)
    prior_array = np.array([1 - prior, prior])
    DCFu=np.sum(expected_cost_per_class * prior_array)
    return DCFu

def DCF_norm(predictions,L,prior,Cfn,Cfp):
    B_dummy=min(prior*Cfn,(1-prior)*Cfp)
    dcf_u=DCF_u(predictions,L,prior,Cfn,Cfp)
    return dcf_u/B_dummy

def get_bayes_decision(llr,pi,Cfn,Cfp):
    pi_Ht=pi
    pi_Hf=1-pi
    t = -np.log((pi_Ht * Cfn) / (pi_Hf * Cfp))
    predictions=np.where(llr>t,1,0)

    return predictions

def compute_min_dcf(llr, L, pi, Cfn, Cfp):
    # Ordinamento degli score (LLR) per ricavare s_1...s_M
    sorted_llr = np.sort(llr)
    
    # Creazione del vettore delle soglie includendo gli estremi -inf e +inf
    thresholds = np.concatenate([np.array([-np.inf]), sorted_llr, np.array([np.inf])])
    
    # Inizializzazione del minimo a un valore infinitamente grande
    min_dcf = np.inf
    
    # Iterazione su tutte le possibili soglie t
    for t in thresholds:
        # Calcolo delle predizioni basate sulla soglia corrente
        predictions = np.where(llr > t, 1, 0)

        # Calcolo della DCF normalizzata
        dcf_n = DCF_norm(predictions,L,pi,Cfn,Cfp)
        
        # Aggiornamento del valore minimo se la soglia corrente risulta più performante
        if dcf_n < min_dcf:
            min_dcf = dcf_n
            
    return min_dcf


def compute_model_llrs(DTR, LTR, DVAL):
    S_MVG = getS(DVAL, get_params_MVG(DTR, LTR))
    S_TiedC = getS_Tied_Covariance(DVAL, get_params_Tied_Covariance(DTR, LTR))
    S_Naive = getS(DVAL, get_params_Naive_Bayes(DTR, LTR))
    return {
        "MVG": S_MVG[1] - S_MVG[0],
        "Tied": S_TiedC[1] - S_TiedC[0],
        "Naive": S_Naive[1] - S_Naive[0],
    }


def print_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def run_application_table(predictions, LVAL):
    applications = [
        (0.5, 1.0, 1.0),
        (0.9, 1.0, 1.0),
        (0.1, 1.0, 1.0),
        (0.5, 1.0, 9.0),
        (0.5, 9.0, 1.0),
    ]

    print(f"{'pi1':>6} {'Cfn':>6} {'Cfp':>6} {'pi_eff':>8} {'DCF_u':>10} {'DCF_norm':>10}")
    for prior, cfn, cfp in applications:
        eff_prior = (prior * cfn) / (prior * cfn + (1 - prior) * cfp)
        dcf_u_value = DCF_u(predictions, LVAL, prior, cfn, cfp)
        dcf_norm_value = DCF_norm(predictions, LVAL, prior, cfn, cfp)
        print(f"{prior:6.1f} {cfn:6.1f} {cfp:6.1f} {eff_prior:8.3f} {dcf_u_value:10.4f} {dcf_norm_value:10.4f}")


def run_effective_prior_table(models, LVAL):
    effective_priors = [0.1, 0.5, 0.9]
    print(f"{'Model':<10} {'pi_eff':>8} {'actualDCF':>12} {'minDCF':>10}")
    for model_name, model_llr in models.items():
        for pi_eff in effective_priors:
            predictions = get_bayes_decision(model_llr, pi_eff, 1, 1)
            actual_dcf = DCF_norm(predictions, LVAL, pi_eff, 1, 1)
            min_dcf = compute_min_dcf(model_llr, LVAL, pi_eff, 1, 1)
            print(f"{model_name:<10} {pi_eff:8.1f} {actual_dcf:12.4f} {min_dcf:10.4f}")


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

        plt.plot(effPriorLogOdds, dcf, label=f"DCF {model_name}")
        plt.plot(effPriorLogOdds, mindcf, label=f"min DCF {model_name}")

    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    D, L, labels = init()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    print_section("1) BASELINE CLASSIFICATION (MVG)")
    predictions_mvg = MVG_Classifier(DTR, LTR, DVAL, LVAL)

    print_section("2) DCF SU APPLICAZIONI (pi1, Cfn, Cfp)")
    run_application_table(predictions_mvg, LVAL)

    print_section("3) CALCOLO LLR DEI MODELLI")
    models = compute_model_llrs(DTR, LTR, DVAL)
    print("LLR pronti per: " + ", ".join(models.keys()))

    print_section("4) CONFRONTO actualDCF vs minDCF")
    run_effective_prior_table(models, LVAL)

    print_section("5) BAYES ERROR PLOT")
    plot_bayes_error(models, LVAL)
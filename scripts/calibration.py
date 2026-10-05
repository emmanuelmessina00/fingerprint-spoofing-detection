import numpy as np
from scripts.logisticreg import *
from scripts.gaussianclassifiers import *
from scripts.logisticreg import *
from scripts.SVM import *
from scripts.GMM import *
def extract_kfold_calibrated_scores(S, L, pi_T, KFOLD):
    pooled_scores = []
    pooled_labels = []
    target_log_odds = np.log(pi_T / (1 - pi_T))

    # Controlliamo se stiamo facendo Fusione (2D) o singola Calibrazione (1D) (Servirà dopo)
    is_fusion = S.ndim > 1

    for idx in range(KFOLD):

        if is_fusion:
            SCAL = np.hstack([S[:, j::KFOLD] for j in range(KFOLD) if j != idx])
            SVAL = S[:, idx::KFOLD]
            SCAL_TRAIN = SCAL  # È già 2D
        else:
            SCAL = np.hstack([S[j::KFOLD] for j in range(KFOLD) if j != idx])
            SVAL = S[idx::KFOLD]
            SCAL_TRAIN = SCAL.reshape(1, -1)

        LCAL = np.hstack([L[j::KFOLD] for j in range(KFOLD) if j != idx])
        LVAL = L[idx::KFOLD]

        v_opt, _ = weighted_trainLogReg(SCAL_TRAIN, LCAL, 0.0, pi_T)

        if is_fusion:
            alpha = v_opt[0:-1].reshape(-1, 1)  # Pesi multipli
            gamma = v_opt[-1] - target_log_odds
            calibrated_SVAL = np.dot(alpha.T, SVAL) + gamma
        else:
            alpha = v_opt[0]  # Peso singolo
            gamma = v_opt[1] - target_log_odds
            calibrated_SVAL = alpha * SVAL + gamma

        pooled_scores.append(calibrated_SVAL.ravel())
        pooled_labels.append(LVAL)

    return np.hstack(pooled_scores), np.hstack(pooled_labels)
def best_LR(DTR,LTR,DVAL):
    DTR_quad = expand_features(DTR)
    DVAL_quad = expand_features(DVAL)
    pi_T=0.1
    target_prior_log_odds=np.log(pi_T/(1-pi_T))
    l=0.03162277660168379
    v_opt,J_min=trainLogReg(DTR_quad,LTR,l)
    w_opt=v_opt[:-1]
    b_opt=v_opt[-1]
    S_val=np.dot(w_opt.T,DVAL_quad)+b_opt
    LP=(S_val>0).astype(int)
    score_lr_quad=S_val-target_prior_log_odds
    return score_lr_quad

def best_SVM(DTR,LTR,DVAL):
    d=2
    xi=1
    pi_T=0.1
    K=0
    c=31.622776601683793
    g=0.1353352832366127
    ZTR=LTR*2-1
    alpha=train_kernel_KRB(K, c, DTR, LTR, xi, g)
    reg_kernel=regular_kernel(KRB(DTR, DVAL, g), xi)
    score_svm_poly=alpha*ZTR @ reg_kernel
    return score_svm_poly

def best_GMM(DTR,LTR,DVAL):
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]
    gmm0 = LBG_constrained(DTR0, threshold=1e-6, target_components=8, alpha=0.1, psi=0.01)
    gmm1 = LBG_constrained(DTR1, threshold=1e-6, target_components=8, alpha=0.1, psi=0.01)
    _, logdens0 = logpdf_GMM(DVAL, gmm0)
    _, logdens1 = logpdf_GMM(DVAL, gmm1)
    llr = logdens1 - logdens0
    return llr

def calibrate_scores(SCAL, LCAL, SVAL, pi_train, pi_target):
    SCAL_2D = SCAL.reshape(1, -1) # otteniamo (1,N) per darlo in pasto alla funzione logReg

    # Addestriamo la LR usando il pi_train ottimale trovato in validazione
    l_reg = 0.0
    v_opt, _ = weighted_trainLogReg(SCAL_2D, LCAL, l_reg, pi_train)

    alpha = v_opt[0] # w
    beta = v_opt[1]  # b

    # Calcoliamo il target_log_odds dinamicamente usando il prior dell'applicazione
    target_log_odds = np.log(pi_target / (1 - pi_target))
    gamma = beta - target_log_odds

    calibrated_SVAL = alpha * SVAL + gamma

    return calibrated_SVAL, alpha, gamma

if __name__ == '__main__':
    pass
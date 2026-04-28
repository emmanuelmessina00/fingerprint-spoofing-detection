import numpy as np
import scipy.optimize

from lda import compute_Sb_Sw, split_db_2to1
from plots import getC, getMu, init, vCol, vRow
from dcf import get_confusion_mat,DCF_u,DCF_norm,get_bayes_decision,compute_min_dcf


def trainLogReg(DTR, LTR, l):
	ZTR = 2 * LTR - 1

	def logreg_obj(v):
		w, b = v[0:-1], v[-1]
		w = w.reshape(-1, 1)
		S = (np.dot(w.T, DTR) + b).ravel()
		loss_terms = np.logaddexp(0, -ZTR * S)
		J = (l / 2) * np.linalg.norm(w) ** 2 + np.mean(loss_terms)
		G = -ZTR / (1.0 + np.exp(ZTR * S))
		grad_w = l * w.ravel() + np.mean(G * DTR, axis=1)
		grad_b = np.mean(G)
		v_grad = np.hstack([grad_w, grad_b])
		return J, v_grad

	x0 = np.zeros(DTR.shape[0] + 1)
	xf, f_min, _ = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0=x0, approx_grad=False)
	return xf, f_min


def weighted_trainLogReg(DTR, LTR, l, piT):
	ZTR = 2 * LTR - 1

	def weighed_logreg_obj(v):
		w, b = v[0:-1], v[-1]
		w = w.reshape(-1, 1)

		S = (np.dot(w.T, DTR) + b).ravel()
		nT = (LTR == 1).sum()
		nF = (LTR == 0).sum()
		xi = np.where(ZTR == 1, piT / nT, (1 - piT) / nF)

		weighed_loss_terms = xi * np.logaddexp(0, -ZTR * S)
		J = (l / 2) * np.linalg.norm(w) ** 2 + np.sum(weighed_loss_terms)

		G = -ZTR / (1.0 + np.exp(ZTR * S))
		grad_w = l * w.ravel() + np.sum(xi * G * DTR, axis=1)
		grad_b = np.sum(xi * G)
		v_grad = np.hstack([grad_w, grad_b])

		return J, v_grad

	x0 = np.zeros(DTR.shape[0] + 1)
	xf, f_min, _ = scipy.optimize.fmin_l_bfgs_b(func=weighed_logreg_obj, x0=x0, approx_grad=False)
	return xf, f_min


def expand_features(D):
	n_features = D.shape[0]
	n_samples = D.shape[1]
	expanded_D = np.zeros((n_features ** 2 + n_features, n_samples))

	for i in range(n_samples):
		x = D[:, i:i + 1]
		xxT = np.dot(x, x.T)
		phi_x = np.vstack([xxT.reshape(-1, 1), x])
		expanded_D[:, i:i + 1] = phi_x

	return expanded_D

if __name__ == "__main__":
	D, L, labels = init()
	(DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

	lambdas=np.logspace(-4,2,13)
pi_T=0.1
target_prior_log_odds=np.log(pi_T/(1-pi_T))

for l in lambdas:
    v_opt,J_min=trainLogReg(DTR,LTR,l)
    w_opt=v_opt[:-1]
    b_opt=v_opt[-1]
    S_val=np.dot(w_opt.T,DVAL)+b_opt
    LP=(S_val>0).astype(int)
    error_rate=np.mean(LP!=LVAL)
    llr=S_val-target_prior_log_odds

    min_dcf=compute_min_dcf(llr,LVAL,pi_T,1,1)
    predictions_bayes=get_bayes_decision(llr,pi_T,1,1)
    act_dcf=DCF_norm(predictions_bayes,LVAL,pi_T,1,1)
    print(f"Lambda: {l}")
    print(f"J ottima: {J_min:e}")
    print(f"Error Rate: {error_rate * 100:.1f}%")
    print(f"DCF min: {min_dcf}")
    print(f"Actual DCF:{act_dcf}\n")
	

lambdas=np.logspace(-4,2,13)
pi_T=0.1
target_prior_log_odds=np.log(pi_T/(1-pi_T))

for l in lambdas:
    v_opt,J_min=trainLogReg(DTR[:,::50],LTR[::50],l)
    w_opt=v_opt[:-1]
    b_opt=v_opt[-1]
    S_val=np.dot(w_opt.T,DVAL)+b_opt
    LP=(S_val>0).astype(int)
    error_rate=np.mean(LP!=LVAL)
    llr=S_val-target_prior_log_odds

    min_dcf=compute_min_dcf(llr,LVAL,pi_T,1,1)
    predictions_bayes=get_bayes_decision(llr,pi_T,1,1)
    act_dcf=DCF_norm(predictions_bayes,LVAL,pi_T,1,1)
    print(f"Lambda: {l}")
    print(f"J ottima: {J_min:e}")
    print(f"Error Rate: {error_rate * 100:.1f}%")
    print(f"DCF min: {min_dcf}")
    print(f"Actual DCF:{act_dcf}\n")


lambdas=np.logspace(-4,2,13)
pi_T=0.1
target_prior_log_odds=np.log(pi_T/(1-pi_T))

for l in lambdas:
    v_opt,J_min=weighted_trainLogReg(DTR,LTR,l,pi_T)
    w_opt=v_opt[:-1]
    b_opt=v_opt[-1]
    
    S_val=np.dot(w_opt.T,DVAL)+b_opt
    LP=(S_val>0).astype(int)
    error_rate = np.mean(LP != LVAL)
    llr=S_val-target_prior_log_odds
    min_dcf=compute_min_dcf(llr,LVAL,pi_T,1,1)

    predictions_bayes = get_bayes_decision(llr, pi_T, 1, 1)
    act_dcf = DCF_norm(predictions_bayes, LVAL, pi_T, 1, 1)

    print(f"Lambda: {l}")
    print(f"J ottima: {J_min:e}")
    print(f"Error Rate: {error_rate * 100:.1f}%")
    print(f"DCF min: {min_dcf}")
    print(f"Actual DCF:{act_dcf}\n")
	
DTR_quad = expand_features(DTR)
DVAL_quad = expand_features(DVAL)

lambdas=np.logspace(-4,2,13)
pi_T=0.1
target_prior_log_odds=np.log(pi_T/(1-pi_T))

for l in lambdas:
    v_opt,J_min=trainLogReg(DTR_quad,LTR,l)
    w_opt=v_opt[:-1]
    b_opt=v_opt[-1]
    S_val=np.dot(w_opt.T,DVAL_quad)+b_opt
    LP=(S_val>0).astype(int)
    error_rate=np.mean(LP!=LVAL)
    llr=S_val-target_prior_log_odds

    min_dcf=compute_min_dcf(llr,LVAL,pi_T,1,1)
    predictions_bayes=get_bayes_decision(llr,pi_T,1,1)
    act_dcf=DCF_norm(predictions_bayes,LVAL,pi_T,1,1)
    print(f"Lambda: {l}")
    print(f"J ottima: {J_min:e}")
    print(f"Error Rate: {error_rate * 100:.1f}%")
    print(f"DCF min: {min_dcf}")
    print(f"Actual DCF:{act_dcf}\n")
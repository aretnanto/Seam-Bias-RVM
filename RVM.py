import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

def random_traintest(n, X, y):
    '''
    Randomly selects n test 
    '''
    idx_test = np.random.choice(int(len(y)), int(n), replace=False)
    idx_train = [list(set(np.arange(len(y))) - set(idx_test))]
    train_x = X[idx_train]
    test_x = X[idx_test]
    train_y = y[idx_train]
    test_y = y[idx_test]
    return train_x, train_y, test_x, test_y


def GaussianKernel(X1, X2,sigma):
    K = np.zeros((len(X1),len(X2)))
    for i in range(0, len(X1)):
        for j in range(0, len(X2)):
            euclid = np.sum((X1[i,:] - X2[j,:])**2)
            K[i,j] = np.exp(-euclid/(2*sigma))
    return K

def RVM(X_train, y, sigma, max_step = 1000, run_kernel = False, stop_s = 1e-6, alpha_init = None, alpha_prune = 1e6, intercept = None, ):
    N = len(y)
    if alpha_init is None: 
        alpha_init = np.ones(N)
    if run_kernel: 
        K = GaussianKernel(X_train, X_train, sigma)
    else:
        K = pd.read_csv('gaussian_kernel.csv', header=None)
    KTK = K.T @ K
    alpha_prior = alpha_init 
    sigma_loop = sigma
    prune_idx = {}
    for i in range(0, max_step):
        delta = (1/sigma_loop)*KTK
        A = np.diag(alpha_prior)
        delta = delta + A
        delta = np.linalg.inv(delta)
        mu = (1/sigma_loop)*delta @ K.T @ y
        alpha_posterior = alpha_prior
        gamma_sum = 0
        for j in range(0,len(alpha_prior)):
            cur_gamma = 1 - alpha_prior[j]*delta[j,j]
            alpha_posterior[j] = cur_gamma/(mu[j])
            gamma_sum += cur_gamma
        prune_safe = alpha_posterior < alpha_prune
        prune_idx[i] = prune_safe
        if sum(prune_safe) != len(alpha_posterior):
            alpha_prior = alpha_prior[prune_safe]
            alpha_posterior = alpha_posterior[prune_safe]
            K = K[:, prune_safe]
            mu = mu[prune_safe]
            delta = delta[prune_safe, :]
            delta = delta[:, prune_safe]
            KTK = K.T @ K
        sigma_loop = np.linalg.norm(y - K @ mu)/(N - gamma_sum)
        if (np.linalg.norm(alpha_posterior - alpha_prior)) < stop_s:
            return mu, sigma_loop, delta, prune_idx
        alpha_prior = alpha_posterior
    return mu, sigma_loop, delta, prune_idx

data = pd.read_csv('sipp.csv')
data = data.loc[(data['RIN_UNIV'] == 1) & (data['TPTOTINC'] > 0)]
data = data[data['EJB1_TYPPAY1'].notna()] # change label as well 
data['DIFF_TPTOTINC'] = data['TPTOTINC'].diff() #assuming data is sorted
mask = data['SSUID'] != data['SSUID'].shift(1) # if it's different unit we don't want to take difference
data['DIFF_TPTOTINC'][mask] = 0
data['MEAN_WAVE'] = data.groupby(['SSUID','SWAVE']).TPTOTINC.transform('mean')
data['MEAN_WAVE'] = (data['MEAN_WAVE'] - np.mean(data['MEAN_WAVE']))/data['MEAN_WAVE'].std()
data['DIFF_TPTOTINC'] = (data['DIFF_TPTOTINC'] - np.mean(data['DIFF_TPTOTINC']))/data['DIFF_TPTOTINC'].std()
data['EMPLOYED'] = data.apply(lambda x: 1 if x['RMESR'] <= 2 else 0, axis=1)
train_data = data.loc[(data['MONTHCODE'] == 12) | (data['MONTHCODE'] == 11) | (data['MONTHCODE'] == 1)]

#Set Train Data
d = train_data['SEAM'].to_numpy().reshape(len(train_data),1)
z = train_data[['MEAN_WAVE','DIFF_TPTOTINC']].to_numpy()

#LS
w_ls = np.linalg.inv(z.T@z) @ z.T @ d
pred_ls = z @ w_ls
pred_ls = np.array([(lambda l: 1 if l >= 0.5 else 0 )(l) for l in pred_ls])
print("LS Seam Bias Detected: " + str(np.sum(pred_ls)))

#RVM and Prume
print("Training RVM")
mu, sigma_loop, delta, prune_idx = RVM(z, d, z.std(), run_kernel = True)
print("Training RVM Done")
print("Kernalizing all data, this might take a while")
K = GaussianKernel(z, z, z.std())
print("Kernalizing done")
pruneK = K
for i in range(0, len(prune_idx)):
    pruneK = pruneK[prune_idx[i]]
pred_rvm_prob = (mu.T @ pruneK).T
pred_rvm = np.array([(lambda l: 1 if l >= 0.5 else 0 )(l) for l in pred_rvm_prob])
print("LS Seam Bias Detected: " + str(np.sum(pred_rvm)))

#Test for Next Stage
z_all = data[['MEAN_WAVE','DIFF_TPTOTINC']].to_numpy()
Kpred = GaussianKernel(z,z_all,z.std())
pruneKpred = Kpred
for i in range(0, len(prune_idx)):
    pruneKpred = pruneKpred[prune_idx[i]]
pred_rvm_prob = (mu.T @ pruneKpred).T
data['SEAM_LS'] = z_all @ w_ls
data['SEAM_BIAS_PROB'] = pred_rvm_prob
data['SEAM_KRR'] = Kpred.T @ np.linalg.inv(K + 1e-12*np.eye(len(K))) @ d

#Train Test Split + Upsample
data_majority = data[data.EMPLOYED==1]
data_minority = data[data.EMPLOYED==0]
data_minority_upsampled = resample(data_minority, 
                                 replace=True,     
                                 n_samples=len(data_majority),    
                                 random_state= 0)
data_up = pd.concat([data_majority, data_minority_upsampled])

#Probability from RVM
X = data_up[['DIFF_TPTOTINC','MEAN_WAVE', 'SEAM_BIAS_PROB']].to_numpy()
y = data_up['EMPLOYED'].to_numpy().reshape((len(data_up),1))
RVM_error = []
for i in [0, 253, 1738, 90210]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    ypred = X_test @ np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    ypred_label = np.array([(lambda l: 1 if l >= 0.5 else 0 )(l) for l in ypred]).reshape(len(ypred),1)
    RVM_error.append(np.sum(ypred_label != y_test)/len(y_test))
print("Mean Error using RVM output:" + str(np.mean(RVM_error)))

#Retaining Orthogonal Components
X = data_up[['DIFF_TPTOTINC','MEAN_WAVE', 'SEAM_LS']].to_numpy()
y = data_up['EMPLOYED'].to_numpy().reshape((len(data_up),1))
LS_error = []
for i in [0, 253, 1738, 90210]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    ypred = X_test @ np.linalg.inv(X_train.T @ X_train + 1e-12*np.eye(len(X_train.T))) @ X_train.T @ y_train
    ypred_label = np.array([(lambda l: 1 if l >= 0.5 else 0 )(l) for l in ypred]).reshape(len(ypred),1)
    LS_error.append(np.sum(ypred_label != y_test)/len(y_test))
print("Mean Error using LS output:" + str(np.mean(LS_error)))


#Gaussian Kernel Ridge
X = data_up[['DIFF_TPTOTINC','MEAN_WAVE', 'SEAM_KRR']].to_numpy()
y = data_up['EMPLOYED'].to_numpy().reshape((len(data_up),1))
KRR_error = []
for i in [0, 253, 1738, 90210]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    ypred = X_test @ np.linalg.inv(X_train.T @ X_train + 1e-12*np.eye(len(X_train.T))) @ X_train.T @ y_train
    ypred_label = np.array([(lambda l: 1 if l >= 0.5 else 0 )(l) for l in ypred]).reshape(len(ypred),1)
    KRR_error.append(np.sum(ypred_label != y_test)/len(y_test))
print("Mean Error using KRR output:" + str(np.mean(KRR_error)))


#Dummy
X = data_up[['DIFF_TPTOTINC','MEAN_WAVE', 'SEAM']].to_numpy()
y = data_up['EMPLOYED'].to_numpy().reshape((len(data_up),1))
Dummy_error = []
for i in [0, 253, 1738, 90210]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    ypred = X_test @ np.linalg.inv(X_train.T @ X_train + 1e-12*np.eye(len(X_train.T))) @ X_train.T @ y_train
    ypred_label = np.array([(lambda l: 1 if l >= 0.5 else 0)(l) for l in ypred]).reshape(len(ypred),1)
    Dummy_error.append(np.sum(ypred_label != y_test)/len(y_test))
print("Mean Error using Dummy output:" + str(np.mean(Dummy_error)))
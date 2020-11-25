import numpy as np
import scipy.stats as sp
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

import matplotlib.pyplot as plt
import seaborn as sb


def x_gen(nt=250,r=2):
    nc = int(r*nt)
    mu = [.2,.2,.5,.5]
    z = np.concatenate([np.ones(nt),np.zeros(nc)])
    
    X = np.zeros((nt+nc,8))
    X[z==0,:4] = np.random.normal(size=(nc,4))
    X[z==1,:4] = np.random.multivariate_normal(mean=mu,cov=np.eye(4),size=(nt))
    X[:,4] = np.random.binomial(1,0.1+0.068*z,size=(nt+nc))
    X[:,5] = np.random.binomial(1,0.1+0.068*z,size=(nt+nc))
    X[:,6] = np.random.binomial(1,0.4+0.242*z,size=(nt+nc))
    X[:,7] = np.random.binomial(1,0.4+0.242*z,size=(nt+nc))

    return X,z

def y_gen(X,z,tau=1,err_std=1):
    nt = int(z.sum())
    nc = int((1-z).sum())

    f1 = 3.5*X[:,0]+4.5*X[:,2]+1.5*X[:,4]+2.5*X[:,6]
    f2 = f1 + 2.5*np.sign(X[:,0])*np.sqrt(abs(X[:,0]))+5.5*X[:,2]**2
    f3 = f2 + 2.5*X[:,2]*X[:,6]-4.5*np.abs(X[:,0]*X[:,2]**3)

    y1 = f1 + tau*z + err_std*np.random.normal(size=(nt+nc,))
    y2 = f2 + tau*z + err_std*np.random.normal(size=(nt+nc,))
    y3 = f3 + tau*z + err_std*np.random.normal(size=(nt+nc,))

    return y1, y2, y3


def tau_diff(y,w,wts=None):
    if wts is None:
        wts = np.ones(y.shape[0])

    y = y.flatten()  

    n1 = (w*wts).sum()
    n0 = ((1-w)*wts).sum()

    ybar1 = (y[w==1]*wts[w==1]).sum()/n1
    ybar0 = (y[w==0]*wts[w==0]).sum()/n0
    tauhat = ybar1 - ybar0

    return tauhat


def cov_mdiff(x,w,wts=None):
    if wts is None:
        wts = np.ones(x.shape[0])
    
    n1 = (w*wts).sum()
    n0 = ((1-w)*wts).sum()

    # weigthed mean for each treatment
    xbar1 = (x[w==1,:].T*wts[w==1]).sum(axis=1)/n1
    xbar0 = (x[w==0,:].T*wts[w==0]).sum(axis=1)/n0
    delta = xbar1 - xbar0

    return delta

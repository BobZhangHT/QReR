import numpy as np
import pandas as pd
import rpy2
from rpy2 import robjects as ro
from rpy2.robjects import Formula
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, pandas2ri

numpy2ri.activate()
pandas2ri.activate()

try:
    stats = importr("stats")
except:
    utils = importr('utils')
    utils.install_packages('stats', repos='http://cran.us.r-project.org')
    stats = importr("stats")

try:
    matching = importr("Matching")
except:
    utils = importr('utils')
    utils.install_packages('Matching', repos='http://cran.us.r-project.org')
    matching = importr("Matching")

def glm_wrapper(x,z):
    columns = ['x'+str(i+1) for i in range(x.shape[1])] + ['z']
    formula = columns[-1]+'~'+'+'.join(columns[:-1])

    kwargs = {"formula": Formula(formula),
              "family": stats.binomial,
              "data": pd.DataFrame(np.hstack([x, z.reshape(-1,1)]),
                                    columns=columns)}

    glm_ps = stats.glm(**kwargs)
    ps_score = glm_ps[2]
    
    return glm_ps, ps_score
    
def match_wrapper(y,z,X):
    kwargs = {'Y':y,
          'Tr':z,
          'X':X,
          'estimand':'ATE',
          'M':1}
    rr = matching.Match(**kwargs)
    tauhat = rr[0]
    se = rr[1]
    lb = tauhat - stats.qnorm(0.975)*se
    ub = tauhat + stats.qnorm(0.975)*se
    return tauhat.item(), lb.item(), ub.item()

def ipw1_wrapper(y,z,x):
    # point estimation
    glm_ps, ps_score = glm_wrapper(x,z)
    tauhat = np.mean(z*y/ps_score)-np.mean((1-z)*y/(1-ps_score))

    # the se calculation follows Lunceford and Davidian (2004)
    H = np.mean((z*y*(1-ps_score)/ps_score-(1-z)*y*ps_score/(1-ps_score)).reshape(-1,1)*x,axis=0)
    ww = ps_score*(1-ps_score)
    E = np.matmul(x.T,ww.reshape(-1,1)*x)/x.shape[0]
    HEinv = np.matmul(H.T,np.linalg.pinv(E))
    xHEinv = np.matmul(x,HEinv)
    se = np.sum((z*y/ps_score-(1-z)*y/(1-ps_score)-tauhat-(z-ps_score)*xHEinv)**2)/(x.shape[0]**2)

    # confidence interval
    qnorm = stats.qnorm(0.975).item()
    lb = tauhat - qnorm*se
    ub = tauhat + qnorm*se
    
    return tauhat, lb, ub
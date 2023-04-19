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
    
# EBCW
try:
    ate = importr("ATE")
except:
    # ATE has been removed from CRAN recently, use `remotes` to install it
    # utils = importr('utils')
    # utils.install_packages('ATE', repos='http://cran.us.r-project.org')
    remotes = importr('remotes')
    remotes.install_version('ATE',version='0.2.0')
    ate = importr("ATE")
    
# EBAL
try:
    weightit = importr("WeightIt")
    survey = importr('survey')
except:
    utils = importr('utils')
    utils.install_packages('WeightIt', repos='http://cran.us.r-project.org')
    utils.install_packages('survey', repos='http://cran.us.r-project.org')
    utils.install_packages('ebal', repos='http://cran.us.r-project.org')
    weightit = importr("WeightIt")
    survey = importr('survey')
    
# SBW (stable weighting)
try:
    optweight = importr("optweight")
    survey = importr('survey')
except:
    utils = importr('utils')
    utils.install_packages('optweight', repos='http://cran.us.r-project.org')
    utils.install_packages('survey', repos='http://cran.us.r-project.org')
    optweight = importr("optweight")
    survey = importr('survey')
    
# FM
try:
    stats = importr('stats')
    matchit = importr("MatchIt")
    sandwich = importr("sandwich")
    lmtest = importr("lmtest")
except:
    utils = importr('utils')
    utils.install_packages('MatchIt', repos='http://cran.us.r-project.org')
    utils.install_packages('sandwich', repos='http://cran.us.r-project.org')
    utils.install_packages('lmtest', repos='http://cran.us.r-project.org')
    utils.install_packages('optmatch', repos='http://cran.us.r-project.org')
    stats = importr('stats')
    matchit = importr("MatchIt")
    sandwich = importr("sandwich")
    lmtest = importr("lmtest")

def sp_infer(y,z,wts):
    '''
    This function performance inference based on weighted regression (for super-population):
    y: response vector, (n,)
    z: allocation vector, (n,)
    wts: sample weights vector, (n,)
    '''
    
    # the following codes are modified from the inference method of WeightIt
    # https://ngreifer.github.io/WeightIt/articles/WeightIt.html
    survey = importr('survey')
    
    design = survey.svydesign(ids=Formula('~1'),
                     weights=wts,
                     data=pd.DataFrame(np.hstack([y.reshape(-1,1), z.reshape(-1,1)]),columns=['y','z']))
    fit = survey.svyglm(Formula('y~z'), design = design)
    summary_fit = survey.summary_svyglm(fit)[12]
    
    est = summary_fit[1,0]
    se = summary_fit[1,1]
    
    lb = est - 1.96*se
    ub = est + 1.96*se
    
    return est, lb, ub

# propensity score estimation using `glm` in R
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
    
# PSM    
def match_wrapper(y,z,X,Z=None,bias_adj=False):
    if Z is None:
        Z = X
    # add bias-adjustment
    kwargs = {'Y':y,
          'Tr':z,
          'Z':Z,
          'X':X,
          'estimand':'ATE',
          'BiasAdjust':bias_adj,
          'M':1}
    rr = matching.Match(**kwargs)
    tauhat = rr[0]
    #se = rr[1]
    #lb = tauhat - 1.96*se
    #ub = tauhat + 1.96*se
    #return tauhat.item(), lb.item(), ub.item()
    return tauhat.item()

# IPW
def ipw_wrapper(y,z,x):
    # point estimation
    glm_ps, ps_score = glm_wrapper(x,z)
    tauhat = np.mean(z*y/ps_score)-np.mean((1-z)*y/(1-ps_score))

    # the se calculation follows Lunceford and Davidian (2004)
    # consider the intercept term
    # w = np.concatenate([np.ones((x.shape[0],1)),x],axis=1)
    # H = np.mean((z*y*(1-ps_score)/ps_score-(1-z)*y*ps_score/(1-ps_score)).reshape(-1,1)*w,axis=0)
    # ww = ps_score*(1-ps_score)
    # E = np.matmul(w.T,ww.reshape(-1,1)*w)/w.shape[0]
    # HEinv = np.matmul(H.T,np.linalg.pinv(E))
    # wHEinv = np.matmul(w,HEinv)
    # var = np.sum((z*y/ps_score-(1-z)*y/(1-ps_score)-tauhat-(z-ps_score)*wHEinv)**2)/(w.shape[0]**2)
    # se = np.sqrt(var)
    
    # confidence interval
    # lb = tauhat - 1.96*se
    # ub = tauhat + 1.96*se
    
    return tauhat

# EBCW
def ate_wrapper(y,z,x):
    kwargs = {'Y':y,
             'Ti':z,
             'X':x}
    ate_results = ate.ATE(**kwargs)
    ate_summary = ate.summary_ATE(ate_results)
    
    est = ate_summary[1][2,0]
    
    return est

# EBAL
def ebal_wrapper(y,z,x,
                 infer_type='sandwich',
                 n_boot=1000,random_state=0):
    columns = ['x'+str(i+1) for i in range(x.shape[1])] + ['z']
    formula = columns[-1]+'~'+'+'.join(columns[:-1])

    kwargs = {"formula": Formula(formula),
              "data": pd.DataFrame(np.hstack([x, z.reshape(-1,1)]),
                                    columns=columns),
              "method": 'ebal',
              'estimand': 'ATE'}

    weightit_results = weightit.weightit(**kwargs)
    wts = weightit_results[0]
    
    if infer_type=='bootstrap':
        
        est = tau_diff(y,z,wts)
        
        # bootstrapping CI
        np.random.seed(random_state)
        est_array = []
        for i in tqdm(range(n_boot)):
            idx = np.random.choice(y.shape[0],y.shape[0],True)
            kwargs = {"formula": Formula(formula),
                  "data": pd.DataFrame(np.hstack([x[idx,:], z[idx].reshape(-1,1)]),
                                        columns=columns),
                  "method": 'ebal',
                  'estimand': 'ATE'}
            weightit_results = weightit.weightit(**kwargs)
            wts = weightit_results[0]
            est_array.append(tau_diff(y[idx],z[idx],wts))
        est_array = np.array(est_array)
        lb,ub = np.quantile(est_array,[0.025,0.975])
    else:
        # the inference follows the official notebook https://ngreifer.github.io/WeightIt/reference/index.html
        
        design = survey.svydesign(ids=Formula('~1'),
                 weights=wts,
                 data=pd.DataFrame(np.hstack([y.reshape(-1,1), z.reshape(-1,1)]),columns=['y','z']))
        fit = survey.svyglm(Formula('y~z'), design = design)
        summary_fit = survey.summary_svyglm(fit)[12]

        est = summary_fit[1,0]
        
    return est

# Stable Weighting (SBW)
def optweight_wrapper(y,z,x,
                 tol=0.01,
                 infer_type='sandwich',
                 n_boot=1000,random_state=0):
    columns = ['x'+str(i+1) for i in range(x.shape[1])] + ['z']
    formula = columns[-1]+'~'+'+'.join(columns[:-1])

    kwargs = {"formula": Formula(formula),
              "data": pd.DataFrame(np.hstack([x, z.reshape(-1,1)]),
                                    columns=columns),
              "method": 'optweight',
              'tol': tol,
              'estimand': 'ATE'}

    weightit_results = weightit.weightit(**kwargs)
    wts = weightit_results[0]
    
    if infer_type=='bootstrap':
        
        est = tau_diff(y,z,wts)
        
        # bootstrapping CI
        np.random.seed(random_state)
        est_array = []
        for i in tqdm(range(n_boot)):
            idx = np.random.choice(y.shape[0],y.shape[0],True)
            kwargs = {"formula": Formula(formula),
                  "data": pd.DataFrame(np.hstack([x[idx,:], z[idx].reshape(-1,1)]),
                                        columns=columns),
                  "method": cbps_type,
                  'estimand': 'ATE'}
            weightit_results = weightit.weightit(**kwargs)
            wts = weightit_results[0]
            est_array.append(tau_diff(y[idx],z[idx],wts))
        est_array = np.array(est_array)
        lb,ub = np.quantile(est_array,[0.025,0.975])
    else:
        # the inference follows the official notebook https://ngreifer.github.io/WeightIt/reference/index.html
        
        design = survey.svydesign(ids=Formula('~1'),
                 weights=wts,
                 data=pd.DataFrame(np.hstack([y.reshape(-1,1), z.reshape(-1,1)]),columns=['y','z']))
        fit = survey.svyglm(Formula('y~z'), design = design)
        summary_fit = survey.summary_svyglm(fit)[12]

        est = summary_fit[1,0]
        
    return est

# FM
def matchit_wrapper(y,z,x,method='full',distance='glm'):
    
    # the full matching is called from optmatch

    columns = ['x'+str(i+1) for i in range(x.shape[1])] + ['z','y']
    formula = columns[-2]+'~'+'+'.join(columns[:-2])

    kwargs = {
        'formula': Formula(formula),
        "data": pd.DataFrame(np.hstack([x, z.reshape(-1,1),y.reshape(-1,1)]),
                                        columns=columns),
        'distance': distance,
        'method': method,
        'estimand': 'ATE'
    }

    m_out = matchit.matchit(**kwargs)
    m_data = matchit.match_data(m_out)
    
    
    # the estimation follows the official tutorial https://cran.r-project.org/web/packages/MatchIt/vignettes/MatchIt.html
    try:
        fit = stats.lm(Formula('y~z'),data=m_data,weights=m_data['weights'])
    except:
        fit = stats.lm(Formula('y~z'),data=m_data,weights=m_data.rx2['weights'])
    summary_fit = lmtest.coeftest(fit, vcov_= sandwich.vcovCL, cluster = Formula('~subclass'))

    est = summary_fit[1,0]
    
    return est

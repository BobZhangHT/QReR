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
    
# ATE
try:
    ate = importr("ATE")
except:
    utils = importr('utils')
    utils.install_packages('ATE', repos='http://cran.us.r-project.org')
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
    
# CBPS
try:
    cbps = importr("CBPS")
    survey = importr('survey')
except:
    utils = importr('utils')
    utils.install_packages('CBPS', repos='http://cran.us.r-project.org')
    utils.install_packages('survey', repos='http://cran.us.r-project.org')
    cbps = importr("CBPS")
    survey = importr('survey')
    
# SW (stable weighting)
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
    se = rr[1]
    lb = tauhat - 1.96*se
    ub = tauhat + 1.96*se
    return tauhat.item(), lb.item(), ub.item()

def ipw1_wrapper(y,z,x):
    # point estimation
    glm_ps, ps_score = glm_wrapper(x,z)
    tauhat = np.mean(z*y/ps_score)-np.mean((1-z)*y/(1-ps_score))

    # the se calculation follows Lunceford and Davidian (2004)
    # consider the intercept term
    w = np.concatenate([np.ones((x.shape[0],1)),x],axis=1)
    H = np.mean((z*y*(1-ps_score)/ps_score-(1-z)*y*ps_score/(1-ps_score)).reshape(-1,1)*w,axis=0)
    ww = ps_score*(1-ps_score)
    E = np.matmul(w.T,ww.reshape(-1,1)*w)/w.shape[0]
    HEinv = np.matmul(H.T,np.linalg.pinv(E))
    wHEinv = np.matmul(w,HEinv)
    var = np.sum((z*y/ps_score-(1-z)*y/(1-ps_score)-tauhat-(z-ps_score)*wHEinv)**2)/(w.shape[0]**2)
    se = np.sqrt(var)
    
    # confidence interval
    lb = tauhat - 1.96*se
    ub = tauhat + 1.96*se
    
    return tauhat, lb, ub

def dr_wrapper(y,z,x):
    from sklearn.linear_model import LinearRegression
    
    # estimate propensity score
    glm_ps, ps_score = glm_wrapper(x,z)

    # estimate linear model for each treatment group
    lr1 = LinearRegression()
    lr0 = LinearRegression()
    lr1.fit(x[z==1,:],y[z==1])
    lr0.fit(x[z==0,:],y[z==0])

    yhat1 = lr1.predict(x)
    yhat0 = lr0.predict(x)

    # point estimator
    tauhat = np.mean((z*y-(z-ps_score)*yhat1)/ps_score)-np.mean(((1-z)*y+(z-ps_score)*yhat0)/(1-ps_score))

    # compute the se (following Gutman & Rubin SMMR 2015)
    I = (z*y-(z-ps_score)*yhat1)/ps_score-((1-z)*y+(z-ps_score)*yhat0)/(1-ps_score)-tauhat
    se = np.sqrt(np.sum(I**2)/(I.shape[0]**2))

    # confidence interval
    lb = tauhat - 1.96*se
    ub = tauhat + 1.96*se
    
    return tauhat, lb, ub
    
def ate_wrapper(y,z,x):
    kwargs = {'Y':y,
             'Ti':z,
             'X':x}
    ate_results = ate.ATE(**kwargs)
    ate_summary = ate.summary_ATE(ate_results)
    
    est = ate_summary[1][2,0]
    lb = ate_summary[1][2,2]
    ub = ate_summary[1][2,3]
    
    return est, lb, ub

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
        se = summary_fit[1,1]
        lb = est - 1.96*se
        ub = est + 1.96*se
    
    return est, lb, ub

# CBPS
def cbps_wrapper(y,z,x,
                 cbps_type='cbps',
                 infer_type='sandwich',
                 n_boot=1000,random_state=0):
    columns = ['x'+str(i+1) for i in range(x.shape[1])] + ['z']
    formula = columns[-1]+'~'+'+'.join(columns[:-1])

    kwargs = {"formula": Formula(formula),
              "data": pd.DataFrame(np.hstack([x, z.reshape(-1,1)]),
                                    columns=columns),
              "method": cbps_type,
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
        se = summary_fit[1,1]
        lb = est - 1.96*se
        ub = est + 1.96*se
    
    return est, lb, ub

# Stable Weighting
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
        se = summary_fit[1,1]
        lb = est - 1.96*se
        ub = est + 1.96*se
    
    return est, lb, ub

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
    
    fit = stats.lm(Formula('y~z'),data=m_data,weights=m_data['weights'])
    summary_fit = lmtest.coeftest(fit, vcov_= sandwich.vcovCL, cluster = Formula('~subclass'))

    est = summary_fit[1,0]
    se = summary_fit[1,1]
    
    lb = est - 1.96*se
    ub = est + 1.96*se
    
    return est, lb, ub
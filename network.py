import numpy as np
import pandas as pd
import scipy.stats as sp
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from datagen import tau_diff


def ReR(pa,x,nt=None):
    '''
    This function performs the rerandomization.
    pa: acceptance probability
    x: data matrix, (n,d)
    nt: number of treatment units
    '''
    
    # transform the format
    if isinstance(x,torch.Tensor):
        x = x.numpy()
    
    n,d = x.shape
    if nt is None:
        nt = int(n/2)
    nt = int(nt)
    nc = int(n - nt)
      
    a = sp.chi2.ppf(pa,df=d)
    w = torch.Tensor([1]*nt+[0]*nc)[torch.randperm(n)]
    mdist = maha_dist(x,w)

    while mdist>a:
        w = torch.Tensor([1]*nt+[0]*nc)[torch.randperm(n)]
        mdist = maha_dist(x,w)

    return w, mdist


def maha_dist(x,w,wts=None,return_cov=False):
    '''
    This function calculates Mahalanobis distance 
    x: data matrix, (n,d)
    w: allocation vector with the shape (n,)
    wts: sample weights vector with the shape (n,) or sample weights matrix with the shape (M,n)
    '''

    if wts is None:
        wts = torch.ones(x.shape[0])
    elif isinstance(wts,np.ndarray):
        wts = torch.Tensor(wts)

    if isinstance(w,np.ndarray):
        w = torch.Tensor(w)

    if isinstance(x,np.ndarray):
        x = torch.Tensor(x)
    
    if len(wts.shape)==1:
        n1 = (w*wts).sum()
        n0 = ((1-w)*wts).sum()

        # weigthed mean for each treatment
        xbar1 = (x[w==1,:].t()*wts[w==1]).sum(axis=1)/n1
        xbar0 = (x[w==0,:].t()*wts[w==0]).sum(axis=1)/n0
        delta = xbar1 - xbar0

        # weighted covariance
        # weighed pool mean
        mu_wt = (x.t()*wts).sum(axis=1)/wts.sum()
        # centered the data
        q_wt = x-mu_wt
        s = wts.sum()
        # covariance
        cov_wt = s*torch.matmul(q_wt.t(),wts.reshape(-1,1)*q_wt)/(s**2-(wts**2).sum())
        try:
            cov_wt_inv = torch.pinverse(cov_wt)
        except:
            # torch.svd may have convergence issues for GPU and CPU.
            # follow https://github.com/pytorch/pytorch/issues/28293
            cov_wt_inv = torch.pinverse(cov_wt+1e-4*cov_wt.mean()*torch.rand(cov_wt.size()[0],cov_wt.size()[1]))

        # mahalanobis distance
        mdist = n1*n0*(cov_wt_inv.matmul(delta)*delta).sum()/(n1+n0)
        
    elif len(wts.shape)==2: # use tensor method (for network)
        wts_mat = wts
        
        # tensorized mahalanobis distance w.r.t wts

        # vector of effective sample size
        n1_vec = wts_mat.matmul(w)
        n0_vec = wts_mat.matmul(1-w)

        # mean differences between two groups
        xbar1_mat = wts_mat[:,w==1].matmul(x[w==1,:])/n1_vec.reshape(-1,1)
        xbar0_mat = wts_mat[:,w==0].matmul(x[w==0,:])/n0_vec.reshape(-1,1)
        delta_mat = xbar1_mat - xbar0_mat

        # pooled mean
        mu_wt_mat = wts_mat.matmul(x)/wts_mat.sum(axis=1).reshape(-1,1)
        mu_wt_tensor = mu_wt_mat.reshape(1,mu_wt_mat.shape[0],mu_wt_mat.shape[1])
        # x-weighted mean
        q_wt_tensor = x.reshape(1,x.shape[0],-1) - mu_wt_mat.reshape(mu_wt_mat.shape[0],1,-1)
        s_vec = wts.sum(axis=1)
        # batch covariance
        cov_wt_tensor = s_vec.reshape(-1,1,1)*torch.matmul(q_wt_tensor.permute([0,2,1]),
                                     q_wt_tensor*wts_mat.reshape(wts_mat.shape[0],-1,1))/(s_vec**2-(wts**2).sum(axis=1)).reshape(-1,1,1)
        # batch covariance inverse
        try:
            cov_wt_inv_tensor = torch.pinverse(cov_wt_tensor)
        except:
            # torch.svd may have convergence issues for GPU and CPU.
            # follow https://github.com/pytorch/pytorch/issues/28293
            cov_wt_inv_tensor = torch.pinverse(cov_wt_tensor+
                                               1e-4*cov_wt_tensor.mean()*torch.rand(cov_wt_tensor.size()[0],
                                                                                    cov_wt_tensor.size()[1],
                                                                                    cov_wt_tensor.size()[2]))
            
        # mdist
        mdist = (n1_vec*n0_vec/(n1_vec+n0_vec))*(torch.matmul(cov_wt_inv_tensor,                                                           delta_mat.reshape(delta_mat.shape[0],-1,1)).squeeze()*delta_mat).sum(axis=1)

    if return_cov:
        try:
            return mdist,cov_wt
        except:
            return mdist,cov_wt_tensor
    else:
        return mdist


class Generator(nn.Module):
    def __init__(self, w, nwts, ngen, ngpu=0):
        super(Generator, self).__init__()
        '''
        w: allocation vector
        nwts: dimension of sample weightss
        ngen: number of hidden neurons
        '''
        self.ngpu = ngpu
        self.fc1 = nn.Linear(nwts,ngen)
        self.fc2 = nn.Linear(ngen,ngen)
        self.fc3 = nn.Linear(ngen,nwts)
        self.w = w # allocation vector
        self.nc = int((1-self.w).sum())
        self.nt = int(self.w.sum())
        
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        
        # first hidden layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # second hidden layer
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # output layer
        x = self.fc3(x)
        x1 = F.softmax(x[:,:self.nt],dim=1)
        x0 = F.softmax(x[:,self.nt:(self.nc+self.nt)],dim=1)
        x = torch.cat([x1,x0],axis=1)
        
        return x

def init_weights_uniform(m):  
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight,-1/2,1/2)
        m.bias.data.fill_(0)

def init_weights_normal(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight,0,np.sqrt(1/6))
        m.bias.data.fill_(0)

def Mdiff_wts(x,w,wts,return_xmat=False):
    '''
    calculate the (weighted) mean difference for each covariate
    x: data matrix, (n,d)
    w: allocation vector, (n,)
    wts: samples weights matrix, (M,n)
    '''
    wts_mat = wts
    
    # vector of effective sample size
    n1_vec = wts_mat.matmul(w)
    n0_vec = wts_mat.matmul(1-w)

    # mean differences between two groups
    xbar1_mat = wts_mat[:,w==1].matmul(x[w==1,:])/n1_vec.reshape(-1,1)
    xbar0_mat = wts_mat[:,w==0].matmul(x[w==0,:])/n0_vec.reshape(-1,1)
    delta_mat = xbar1_mat - xbar0_mat
    
    if return_xmat is True:
        return delta_mat, xbar1_mat, xbar0_mat
    else:
        return delta_mat

def MdiffLoss(output, target, lam=1):
    '''
    This loss enforces the first two moments of mean differences matrix between the output and target to be the same.
    In the latest version, this loss is just used to show the deviation degree of the first and second moments. 
    output, target: mean difference matrix, (n,d)
    '''
    def cov_mat_cpt(tmp):
        tmp_cen = tmp - tmp.mean(axis=0).reshape(-1,tmp.shape[1])
        cov_mat = tmp_cen.t().matmul(tmp_cen)/(tmp_cen.shape[0]-1)
        return cov_mat

    mean_target = target.mean(axis=0)
    cov_target = cov_mat_cpt(target)

    mean_output = output.mean(axis=0)
    cov_output = cov_mat_cpt(output)

    mean_loss = torch.sum((mean_output-mean_target)**2)
    var_loss = torch.sum((torch.pinverse(cov_target).matmul(cov_output)-torch.eye(cov_output.shape[0]))**2)
    
    return (mean_loss + var_loss), mean_loss, var_loss
    
def KS(output, target):
    '''
    The pytorch codes calculate the KS statistics for two empirical distributions, modified from ks_2samp in scipy.stats.
    '''
    data1 = torch.sort(output.view(-1))[0]
    data2 = torch.sort(target.view(-1))[0]
    n1 = data1.shape[0]
    n2 = data2.shape[0]

    data_all = torch.cat([data1, data2])
    # using searchsorted solves equal data problem
    cdf1 = torch.searchsorted(data1, data_all, right=True) / n1
    cdf2 = torch.searchsorted(data2, data_all, right=True) / n2
    cddiffs = cdf1 - cdf2
    minS = torch.clip(-torch.min(cddiffs), 0, 1)  # ensure sign of minS is not negative.
    maxS = torch.max(cddiffs)
    return torch.max(minS, maxS)

def poly_kernel(x_input,y_input=None,gamma=1,degree=2,c=0):
    if y_input is None:
        gram_mat = torch.matmul(x_input,x_input.t())
    else:
        gram_mat = torch.matmul(x_input,y_input.t())
    return (gamma*gram_mat+c)**degree

def rbf_kernel(x_input,y_input=None,gamma=1):
    if y_input is None:
        rbf_mat = torch.exp(-torch.norm(x_input.unsqueeze(1)-x_input, dim=2, p=2)**2*gamma)
    else:
        rbf_mat = torch.exp(-torch.norm(x_input.unsqueeze(1)-y_input, dim=2, p=2)**2*gamma)
    return rbf_mat

def sigmoid_kernel(x_input,y_input=None,gamma=1,c=0):
    if y_input is None:
        gram_mat = torch.matmul(x_input,x_input.t())
    else:
        gram_mat = torch.matmul(x_input,y_input.t())
    return torch.tanh(gamma*gram_mat+c)

def MMDLoss(x_input,y_input,
            kernel='rbf',
            gamma=1,degree=2,c=1):
    '''
    This loss use the kernel trick to evaluate the deviation of two distributions.
    x_input, y_input: input empirical samples from two distributions, (n1,d1), (n2,d2).
    kernel: types of the kernel.
    gamma, degree, c: parameters of the kernel. 
    '''
    
    if kernel == 'rbf':
        loss = rbf_kernel(x_input,None,gamma).mean()+rbf_kernel(y_input,None,gamma).mean()-2*rbf_kernel(x_input,y_input,gamma).mean()
    elif kernel == 'poly':
        loss = poly_kernel(x_input,None,gamma,degree,c).mean()+poly_kernel(y_input,None,gamma,degree,c).mean()-2*poly_kernel(x_input,y_input,gamma,degree,c).mean()
    elif kernel == 'sigmoid':
        loss = sigmoid_kernel(x_input,None,gamma,c).mean()+sigmoid_kernel(y_input,None,gamma,c).mean()-2*sigmoid_kernel(x_input,y_input,gamma,c).mean()
    elif kernel == 'linear':
        x_gram_mat = torch.matmul(x_input,x_input.t())
        y_gram_mat = torch.matmul(y_input,y_input.t())
        xy_gram_mat = torch.matmul(x_input,y_input.t())
        loss = x_gram_mat.mean()+y_gram_mat.mean()-2*xy_gram_mat.mean()
    return loss.sqrt()


class QReR(BaseEstimator):
    '''
    A scikit-learn wrapper of our proposed method.
    '''
    def __init__(self, 
                 # parameters of training
                 lr=1e-3,
                 batch_size=256,
                 patience=10,
                 num_iters=1500,
                 num_init_iters=1000,
                 num_noval_iters=0,
                 pa=np.inf,
                 verbose=False,
                 random_state=0,
                 save_folder='./save/',
                 # parameters of network
                 x_lambda=1,
                 wt_lambda=0,
                 val_metric='KS',
                 kernel_params=None,
                 num_nodes=64, 
                 ngpu=0, 
                 device='cpu'):
        
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience # patience for early stopping
        self.num_iters = num_iters # number of iterations
        self.num_init_iters = num_init_iters
        self.num_noval_iters = num_noval_iters # iterations that do not consider validation
        self.pa = pa # acceptance probability for the rerandomization
        self.verbose = verbose # whether to show the information
        self.random_state = random_state
        self.save_folder = save_folder
        
        self.x_lambda = x_lambda
        self.wt_lambda = wt_lambda
        self.num_nodes = num_nodes # number of hidden nodes
        self.ngpu = ngpu
        self.device = device
        
        if kernel_params is None:
            self.kernel_params = {'kernel':'poly',
                 'gamma':1,
                 'degree':2,
                 'c':0}
        else:  
            self.kernel_params = kernel_params
        self.val_metric = val_metric
    
    def _noise_gen(self,num_noise=1000):
        
        # this function generate noisy weights from dirichlet distribution
        wts1_dirich = np.random.dirichlet(np.ones(self.nt),size=(num_noise,))
        wts0_dirich = np.random.dirichlet(np.ones(self.nc),size=(num_noise,))
        wts_dirich = np.concatenate([wts1_dirich,wts0_dirich],axis=1)
        
        noise = torch.from_numpy(wts_dirich).float().to(self.device)
        
        return noise
    
    def _wts_gen(self,noise):
        with torch.no_grad():
            # forward pass: compute predicted outputs by passing inputs to the model
            wts = self.netG(noise).cpu().detach()
            # here we multiply with nt & nc to avoid too small wts values, 
            # which may leads to some problems about float precision during calculation
            wts[:,:self.nt] = wts[:,:self.nt]*self.nt 
            wts[:,self.nt:] = wts[:,self.nt:]*self.nc
        return wts
    
    def _check_input(self,x):
        if isinstance(x,np.ndarray):
            return torch.Tensor(x)
        elif isinstance(x,torch.Tensor):
            return x
        else:
            raise TypeError('The instance should be either numpy.ndarray or torch.Tensor, but get {}'.format(type(x)))

    def _init_data(self,x,w_rer_mat=None,
            mdist_rer_vec=None,
            num_fixed_noise=1000,
            num_rer=10000):

        # generate some fixed noise for evaluation
        self.fixed_noise = self._noise_gen(num_fixed_noise)
        
        # generate fixed real mdist and mdiff for evaluation (from ReR)
        fixed_w_rer_mat = []
        self.real_fixed_mdist = []
        for i in range(num_fixed_noise):
            w_rer, mdist = ReR(self.pa,x,self.nt)
            self.real_fixed_mdist.append(mdist)
            fixed_w_rer_mat.append(w_rer.reshape(-1,1))
        
        self.real_fixed_mdist = torch.Tensor(self.real_fixed_mdist)
        fixed_w_rer_mat = torch.cat(fixed_w_rer_mat,axis=1).t().float()
        self.x1_fixed_mat = fixed_w_rer_mat.matmul(x)/fixed_w_rer_mat.sum(axis=1).reshape(-1,1)
        self.x0_fixed_mat = (1-fixed_w_rer_mat).matmul(x)/((1-fixed_w_rer_mat).sum(axis=1).reshape(-1,1))
        self.real_fixed_mdiff = self.x1_fixed_mat - self.x0_fixed_mat
        
        # generate the randomized allocations with mahalanobis distance if they are not provided (from ReR)
        # these allocation vectors are used for network training
        if (w_rer_mat is None) or (mdist_rer_vec is None):
            self.w_rer_mat = []
            self.mdist_rer_vec = []
            for i in range(num_rer):
                w_rer, mdist = ReR(self.pa,x,self.nt)
                self.mdist_rer_vec.append(mdist)
                self.w_rer_mat.append(w_rer.reshape(-1,1))
        
        self.mdist_rer_vec = torch.Tensor(self.mdist_rer_vec)
        self.w_rer_mat = torch.cat(self.w_rer_mat,axis=1).t().float()
        self.x1_mat = self.w_rer_mat.matmul(x)/self.w_rer_mat.sum(axis=1).reshape(-1,1)
        self.x0_mat = (1-self.w_rer_mat).matmul(x)/((1-self.w_rer_mat).sum(axis=1).reshape(-1,1))
        self.x_mdiff_rer = self.x1_mat - self.x0_mat


    def _init_network(self):
        # network definition
        self.netG = Generator(w=self.w, nwts=self.nwts, 
                         ngen=self.num_nodes, ngpu=self.ngpu).to(self.device)
        # rough initialization
        self.netG.apply(init_weights_uniform)
        # optimizer for initialization
        self.init_optimizer = optim.Adam(self.netG.parameters(), lr=self.lr)
        
        input_wts = self._noise_gen()
        
        print('Initialize the network via pretraining.')
        for i in tqdm(range(self.num_init_iters)):
            self.netG.zero_grad()
            output_wts = self.netG(input_wts)
            mse_loss = torch.mean(torch.sum((torch.log(output_wts) - torch.log(input_wts))**2,axis=1))
            mse_loss.backward()
            self.init_optimizer.step()
        print('Pretraining complete!') 

        
    def fit(self,x,w,
            w_rer_mat=None,
            mdist_rer_vec=None,
            num_fixed_noise=1000,
            num_rer=10000):
        
        # data should be preprocessed to ensure that first nt of w is 1 and last nc of w is 0
        # w = [1,1,1,1,...,1,0,0,0,...,0]
        
        x = self._check_input(x)
        w = self._check_input(w)
        
        self.x = x
        self.xbar = torch.mean(self.x,axis=0).unsqueeze_(0)
        self.w = w
        self.nt = int(w.sum().item()) # number of treatment samples
        self.nc = int((1-w).sum().item()) # number of control samples
        self.nwts = int(self.w.shape[0]) # number of samples
        
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        
        # data initialization
        self._init_data(x,w_rer_mat,mdist_rer_vec,num_fixed_noise,num_rer)

        # network initialization
        self._init_network()
        
        # gamma initialization
        self.net_gamma = nn.Sequential(nn.Linear(1,1))
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
        self.net_gamma.apply(weights_init)
        self.init_gamma_val = np.log(self.kernel_params['gamma'])
          
        # setup optimizers for network and kernel gamma
        self.optimizer = optim.Adam(self.netG.parameters(), lr=self.lr)
        self.optimizer_gamma = optim.Adam(self.net_gamma.parameters(), lr=self.lr)
        
        # begin training
        self.losses = [] # training loss
        self.val_losses = [] # validation loss
        self.ks_list = [] # training KS statistics
        self.val_ks_list = [] # validation KS statistics
        self.x_mdiff_list = [] # average absolute mean difference of covariates
        
        self.best_val_metric = np.inf
        stop_cnt = 0

        print("Starting Training Loop...")
        for iteration in range(self.num_iters):
            
            
            ####################
            # update netG
            ####################
            self.netG.zero_grad()
            
            # generate real data (using subsampling)
            real_idx = np.random.choice(num_rer,self.batch_size)
            real_dist = torch.Tensor(self.mdist_rer_vec[real_idx])
            real_mdiff = torch.Tensor(self.x_mdiff_rer[real_idx])

            # generate fake data (from dirichlet distribution)
            noise = self._noise_gen(self.batch_size)
            gen_wts = self.netG(noise)

            # for the generated weights, we multiply the sample size to ensure avoid the possible loss of float precision
            gen_wts[:,:self.nt] = gen_wts[:,:self.nt]*self.nt
            gen_wts[:,self.nt:] = gen_wts[:,self.nt:]*self.nc
            fake_mdiff, fake_xbar1_mat, fake_xbar0_mat = Mdiff_wts(x,w,gen_wts,return_xmat=True)
            fake_dist = maha_dist(x,w,gen_wts)
            
            # loss that measures deviation of first two moments
            _, meanloss, varloss = MdiffLoss(fake_mdiff, real_mdiff)
            
            # MMDLoss
            self.kernel_params['gamma'] = torch.exp(self.net_gamma(torch.tensor([self.init_gamma_val],dtype=torch.float32)))
            mmdloss = MMDLoss(fake_mdiff,real_mdiff,self.kernel_params['kernel'],self.kernel_params['gamma'].detach(),
                             self.kernel_params['degree'],self.kernel_params['c'])
            
            # regularity for the generated weights
            wt_reg_term = torch.sum((gen_wts[:,:self.nt]/self.nt-1/self.nt)**2,axis=1).mean() + \
                            torch.sum((gen_wts[:,self.nt:]/self.nc-1/self.nc)**2,axis=1).mean()
            
            # regularity for the covariate
            x_reg_term = torch.sum((fake_xbar1_mat - self.xbar)**2,axis=1).mean() + torch.sum((fake_xbar0_mat - self.xbar)**2,axis=1).mean()

            errG = mmdloss + self.x_lambda*x_reg_term + self.wt_lambda*wt_reg_term
            ksG = KS(fake_dist, real_dist)
            errG.backward()
            self.optimizer.step()
            
            ####################
            # update gamma
            ####################
            self.net_gamma.zero_grad()
            gen_wts = self.netG(noise)
            gen_wts[:,:self.nt] = gen_wts[:,:self.nt]*self.nt
            gen_wts[:,self.nt:] = gen_wts[:,self.nt:]*self.nc
            fake_mdiff = Mdiff_wts(x,w,gen_wts)
            
            mmdloss = MMDLoss(fake_mdiff.detach(),real_mdiff,self.kernel_params['kernel'],self.kernel_params['gamma'],
                             self.kernel_params['degree'],self.kernel_params['c'])

            err_gamma = -mmdloss
            err_gamma.backward()
            self.optimizer_gamma.step()
            
            ####################
            # saving & early-stopping
            ####################
            # save Losses & KS on training set
            self.losses.append(errG.item())
            self.ks_list.append(ksG.item())
            self.x_mdiff_list.append(fake_mdiff.mean(axis=0).abs().mean().item()) 

            if (iteration+1) % 50 == 0:
                
                # save Losses & KS on validation set
                wts_mat_net = self._wts_gen(self.fixed_noise)
                
                # save the validation ks score
                fake_fixed_mdist, xcov_fixedwt_tensor = maha_dist(x,w,wts_mat_net,True)
                val_ks = KS(fake_fixed_mdist,self.real_fixed_mdist).item()
                self.val_ks_list.append(val_ks)
                
                # save the validation loss
                fake_fixed_mdiff, fake_fixed_xbar1_mat, fake_fixed_xbar0_mat = Mdiff_wts(x,w,wts_mat_net, return_xmat=True)
                
                # no penalty is required during validation
                # https://www.pyimagesearch.com/2019/10/14/why-is-my-validation-loss-lower-than-my-training-loss/ 
                val_loss = MMDLoss(fake_fixed_mdiff,self.real_fixed_mdiff,**self.kernel_params)
                self.val_losses.append(val_loss.item())
                
                
                if self.verbose is True:
                    print('[%d/%d]\tLoss: %.4f (1stMoM:%.4f, 2ndMoM:%.4f, wt_reg_term:%.4f, x_reg_term:%.4f)\tValMMDLoss: %.4f\tKS: %.4f\tValKS: %.4f\tMdiff_Avg: %.4f\tgamma: %.4f\t'
                            % (iteration+1, self.num_iters, 
                               errG.item(),
                               meanloss.item(),
                               varloss.item(),
                               wt_reg_term.item(),
                               x_reg_term.item(),
                               val_loss,
                               ksG.item(),
                               val_ks,
                               fake_mdiff.mean(axis=0).abs().mean().item(),
                               self.kernel_params['gamma'].item()))
                    
                    # show the distribution
                    # fig,axes = plt.subplots(1,2,figsize=(9,4))
                    plt.figure(figsize=(9,4))
                    
                    plt.subplot(121)
                    sb.distplot(self.real_fixed_mdist,label='True')
                    sb.distplot(fake_fixed_mdist,label='Generated')
                    plt.legend()
                    plt.title('KS: '+str(np.round(val_ks,4)))
                    
                    plt.subplot(122)
                    plt.plot(self.val_losses,label='Validation')
                    #plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.show()
                
                if iteration >= self.num_noval_iters:
                    if self.val_metric == 'KS':
                        val_metric_ = val_ks
                    elif self.val_metric == 'Loss':
                        val_metric_ = val_loss

                    if val_metric_ < self.best_val_metric:
                        self.best_val_metric = val_metric_
                        stop_cnt = 0
                    else:
                        stop_cnt += 1

            if stop_cnt >= self.patience:
                print('Early Stop the Training.')
                torch.save(self.netG.state_dict(), self.save_folder+'final_checkpoint.pt')
                self.netG.load_state_dict(torch.load(self.save_folder+'final_checkpoint.pt'))
                break

        if stop_cnt >= self.patience:
            return self
        else:
            print('Training Complete.')
            torch.save(self.netG.state_dict(), self.save_folder+'final_checkpoint.pt')
            return self
    
    def predict(self,num_noise=1000):
        
        check_is_fitted(self, "netG")
        
        noise = self._noise_gen(num_noise)
        wts = self._wts_gen(noise)
        
        return wts
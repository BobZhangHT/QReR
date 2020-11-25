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

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator


def ReR(pa,x):
    '''
    pa: acceptance probability
    x: data matrix
    '''
    
    # transform the format
    if isinstance(x,torch.Tensor):
        x = x.numpy()

    n,d = x.shape
    a = sp.chi2.ppf(pa,df=d)

    w = torch.tensor([0,1]).repeat(int(n/2))[torch.randperm(n)]
    mdist = maha_dist(x,w)

    while mdist>a:
        w = torch.tensor([0,1]).repeat(int(n/2))[torch.randperm(n)]
        mdist = maha_dist(x,w)

    return w, mdist


def maha_dist(x,w,wts=None):
    '''
    x: data matrix
    w: allocation vector
    wts: sample weights
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
        #xbar1 = torch.matmul(x.t(),w*wts)/n1
        #xbar0 = torch.matmul(x.t(),(1-w)*wts)/n0
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
        cov_wt_inv = torch.inverse(cov_wt)

        # mahalanobis distance
        mdist = n1*n0*(cov_wt_inv.matmul(delta)*delta).sum()/(n1+n0)
        
    elif len(wts.shape)==2: # use tensor method (for network)
        wts_mat = wts
        
        # tenorized mahalanobis distance w.r.t wts

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
        cov_wt_inv_tensor = torch.inverse(cov_wt_tensor)
        # mdist
        mdist = (n1_vec*n0_vec/(n1_vec+n0_vec))*(torch.matmul(cov_wt_inv_tensor,                                                           delta_mat.reshape(delta_mat.shape[0],-1,1)).squeeze()*delta_mat).sum(axis=1)

    return mdist


class Generator(nn.Module):
    def __init__(self, w, nwts, ngen, ngpu=0):
        super(Generator, self).__init__()
        '''
        w: allocation vector
        nwts: dimension of sample weights
        ngen: number of features
        '''
        self.ngpu = ngpu
        self.fc1 = nn.Linear(nwts, ngen)
        self.fc2 = nn.Linear(ngen,ngen)
        self.fc3 = nn.Linear(ngen,nwts)
        self.w = w # allocation vector
        self.nc = int((1-self.w).sum())
        self.nt = int(self.w.sum())

    def forward(self, x):
        
        # first
        x = self.fc1(x)
        x = F.relu(x)
        
        # second
        x = self.fc2(x)
        x = F.relu(x)
        
        # third
        x = self.fc3(x)
        
        x1 = F.softmax(x[:,:self.nt],dim=1)
        x0 = F.softmax(x[:,self.nt:(self.nc+self.nt)],dim=1)
        
        x = torch.cat([x1,x0],axis=1)
        return x
    
def QQLoss(output, target):
    '''
    motivated by qq-plot, we define a loss based on quantiles of output and target
    output, target: the mahalanobis distance samples
    '''
    # quantile based loss
    output_qt = torch.quantile(output,torch.Tensor(np.linspace(0,1,99))) 
    target_qt = torch.quantile(target,torch.Tensor(np.linspace(0,1,99))) 

    return torch.sum((output_qt-target_qt)**2)


def Mdiff_wts(x,w,wts):
    '''
    calculate the mean difference for each covariate
    x: data matrix
    w: allocation vector
    wts: samples weights
    '''
    wts_mat = wts
    
    # vector of effective sample size
    n1_vec = wts_mat.matmul(w)
    n0_vec = wts_mat.matmul(1-w)

    # mean differences between two groups
    xbar1_mat = wts_mat[:,w==1].matmul(x[w==1,:])/n1_vec.reshape(-1,1)
    xbar0_mat = wts_mat[:,w==0].matmul(x[w==0,:])/n0_vec.reshape(-1,1)
    delta_mat = xbar1_mat - xbar0_mat
    return delta_mat

def MdiffLoss(output, target):
    '''
    This loss enforces the mean differences matrix between the output and target to be the same.
    We control the batch mean and variance of the covarate mean differences.
    We use log(std) to amplify the differences between variances.
    The scalar is introduced to balance the maganitude between variance and mean component.
    output, target: mean difference matrix (nbatch, ncovariate)
    '''
    mean_target = target.mean(axis=0)
    std_target = target.std(axis=0)

    mean_output = output.mean(axis=0)
    std_output = output.std(axis=0)

    mean_loss = torch.sum((mean_output-mean_target)**2)
    var_loss = torch.sum((torch.log(std_target)-torch.log(std_output))**2)

    scalar = var_loss.item()/mean_loss.item()
    return 2*scalar*mean_loss + var_loss

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
    minS = torch.clip(-torch.min(cddiffs), 0, 1)  # Ensure sign of minS is not negative.
    maxS = torch.max(cddiffs)
    return torch.max(minS, maxS)


class QRWG(BaseEstimator):
    def __init__(self, 
                 # parameters of training
                 lr=1e-3,
                 batch_size=256,
                 patience=10,
                 num_iters=1500,
                 pa=np.inf,
                 verbose=False,
                 random_state=0,
                 save_folder='./save/',
                 # parameters of network
                 num_nodes=64, ngpu=0, device='cpu'):
        
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience # patience for early stopping
        self.num_iters = num_iters # number of iterations
        self.pa = pa # acceptance probability for the rerandomization
        self.verbose = verbose # whether to show the information
        self.random_state = random_state
        self.save_folder = save_folder
        
        self.num_nodes = num_nodes # number of hidden nodes
        self.ngpu = ngpu
        self.device = device
    
    
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
            wts = wts*(self.nt+self.nc)/2
        return wts
    
    def _check_input(self,x):
        if isinstance(x,np.ndarray):
            return torch.Tensor(x)
        elif isinstance(x,torch.Tensor):
            return x
        else:
            raise TypeError('The instance should be either numpy.ndarray or torch.Tensor, but get {}'.format(type(x)))
        
    def fit(self,x,w,
            w_rer_mat=None,
            mdist_rer_vec=None,
            num_fixed_noise=1000,
            num_rer=5000):
        
        x = self._check_input(x)
        w = self._check_input(w)
        
        self.w = w
        self.nt = int(w.sum().item()) # number of treatment samples
        self.nc = int((1-w).sum().item()) # number of control samples
        self.nwts = int(self.w.shape[0]) # number of samples
        
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        
        # network 
        self.netG = Generator(w=self.w, nwts=self.nwts, 
                         ngen=self.num_nodes, ngpu=self.ngpu).to(self.device)
     
        # setup adam optimizers
        optimizer = optim.Adam(self.netG.parameters(), lr=self.lr)
        
        # generate some fixed noise for evaluation
        self.fixed_noise = self._noise_gen(num_fixed_noise)
        
        # generate fixed real mdist for evaluation
        real_fixed_mdist = torch.Tensor([ReR(self.pa,x)[1] for i in range(num_fixed_noise)])
        
        # generate the randomized allocations with mahalanobis distance if they are not provided
        if (w_rer_mat is None) or (mdist_rer_vec is None):
            self.w_rer_mat = []
            self.mdist_rer_vec = []
            for i in range(num_rer):
                w_rer, mdist = ReR(self.pa,x)
                self.mdist_rer_vec.append(mdist)
                self.w_rer_mat.append(w_rer.reshape(-1,1))
        
        # mean difference matrix for covariate
        self.mdist_rer_vec = torch.Tensor(self.mdist_rer_vec)
        self.w_rer_mat = torch.cat(self.w_rer_mat,axis=1).t().float()
        x1_mat = self.w_rer_mat.matmul(x)/self.w_rer_mat.sum(axis=1).reshape(-1,1)
        x0_mat = (1-self.w_rer_mat).matmul(x)/((1-self.w_rer_mat).sum(axis=1).reshape(-1,1))
        self.x_mdiff_rer = x1_mat - x0_mat
        
        # begin training
        self.losses = []
        self.qqlosses = []
        self.mdifflosses = []
        self.ks_list = []
        self.val_ks_list = []
        self.x_mdiff_list = []

        self.best_val_ks = np.inf
        stop_cnt = 0

        print("Starting Training Loop...")
        for iteration in range(self.num_iters):

            self.netG.zero_grad()
            # generate real data
            real_idx = np.random.choice(num_rer,self.batch_size)
            real_dist = torch.Tensor(self.mdist_rer_vec[real_idx])
            real_dist = torch.log(real_dist) # take log to accelerate the convergence
            real_mdiff = torch.Tensor(self.x_mdiff_rer[real_idx])

            # generate fake data
            wts1_dirich = np.random.dirichlet(np.ones(self.nt),size=(self.batch_size,))
            wts0_dirich = np.random.dirichlet(np.ones(self.nc),size=(self.batch_size,))
            wts_dirich = np.concatenate([wts1_dirich,wts0_dirich],axis=1)
            noise = torch.from_numpy(wts_dirich).float().to(self.device)
            gen_wts = self.netG(noise)
            
            # here we multiple (nt+nc/2) to ensure the total sample size as nt+nc with half of them in the treatment group
            fake_mdiff = Mdiff_wts(x,w,gen_wts*(self.nt+self.nc)/2)
            fake_dist = torch.log(maha_dist(x,w,gen_wts*(self.nt+self.nc)/2))

            # backward propagation
            qqloss = QQLoss(fake_dist, real_dist)
            mdiffloss = MdiffLoss(fake_mdiff, real_mdiff)
            errG = 0.5*qqloss + mdiffloss
            ksG = KS(fake_dist, real_dist)
            errG.backward()
            optimizer.step()

            # save Losses & KS for plotting later
            self.losses.append(errG.item())
            self.qqlosses.append(qqloss.item())
            self.mdifflosses.append(mdiffloss.item())
            self.ks_list.append(ksG.item())
            self.x_mdiff_list.append(fake_mdiff.mean(axis=0).abs().mean().item())

            
            if self.verbose is True:
                if (iteration+1) % 10 == 0:
                      print('[%d/%d]\tTotal Loss: %.4f\tMdiffLoss: %.4f\tQQLoss: %.4f\tKS: %.4f\tMdiff_Avg: %.4f\t'
                            % (iteration+1, self.num_iters, 
                               errG.item(),
                               mdiffloss.item(),
                               qqloss.item(),
                               ksG.item(),
                               fake_mdiff.mean(axis=0).abs().mean().item()))

            if (iteration+1) % 50 == 0:
                wts_mat_net = self._wts_gen(self.fixed_noise)
                fake_fixed_mdist = maha_dist(x,w,wts_mat_net)

                # save the validation ks score
                val_ks = KS(fake_fixed_mdist,real_fixed_mdist).item()
                self.val_ks_list.append(val_ks)
                
                if self.verbose is True:
                    # show the distribution
                    plt.figure()
                    sb.distplot(real_fixed_mdist,label='True')
                    sb.distplot(fake_fixed_mdist,label='Generated')
                    plt.legend()
                    plt.title('KS: '+str(np.round(val_ks,4)))
                    plt.show()

                if val_ks < self.best_val_ks:
                    self.best_val_ks = val_ks
                    stop_cnt = 0
                else:
                    stop_cnt += 1
                
                print('Update Model.')
                torch.save(self.netG.state_dict(), self.save_folder+'checkpoint.pt')

                if stop_cnt >= self.patience:
                    print('Stop the Training.')
                    self.netG.load_state_dict(torch.load(self.save_folder+'checkpoint.pt'))
                    break
        return self
    
    def predict(self,num_noise=1000):
        
        check_is_fitted(self, "netG")
        
        np.random.seed(self.random_state)
        
        noise = self._noise_gen(num_noise)
        wts = self._wts_gen(noise)
        
        return wts
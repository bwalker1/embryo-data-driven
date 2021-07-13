""" This file implements some sinkhorn algorithms for computing optimal transport
    in PyTorch based on the Python Optimal Transport library (https://github.com/PythonOT/POT),
    which implements recent OT algorithms.
"""
import ot
import numpy as np
from scipy.spatial import distance_matrix
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

def sinkhorn_knopp(a, b, M, eps, 
             numItermax=1000, stopThr=1e-9, avoid_zero=True):
    if avoid_zero:
        a = a + 1e-9
        a = a / (1.0 + a.shape[0] * 1e-9)
        b = b + 1e-9
        b = b / (1.0 + b.shape[0] * 1e-9)
    eps_inv = 1.0/eps
    M_eps_inv = - M * eps_inv
    K = torch.exp(M_eps_inv)
    Kt = torch.transpose(K, 0, 1)
    Kp = (1.0/a).view(-1,1) * K
    n = a.shape[0]
    m = b.shape[0]
    u = torch.ones(n, dtype=M.dtype, device=M.device) * (1/n)
    v = torch.ones(m, dtype=M.dtype, device=M.device) * (1/m)
    uprev = torch.empty(n, dtype=M.dtype, device=M.device)
    vprev = torch.empty(m, dtype=M.dtype, device=M.device)
    numIter = 0
    err = 1.0
    while (err > stopThr and numIter < numItermax):
        uprev = u + 0.0
        vprev = v + 0.0
        KtU = torch.mv(Kt, u)
        v = b.div(KtU)
        u = 1./torch.mv(Kp, v)
        if  (KtU==0).any() or \
            torch.isnan(u).any() or \
            torch.isnan(v).any() or \
            torch.isinf(u).any() or \
            torch.isinf(v).any():
            u = uprev + 0.0
            v = vprev + 0.0
            break
        if numIter % 10 == 0:
            tmp = torch.einsum('i,ij,j->j', u, K, v)
            err = torch.norm(tmp - b) ** 2
        numIter += 1
    gamma = u.view(-1,1) * K * v.reshape(1,-1)
    # res = torch.einsum('ik,ij,jk,ij->k', u, K, v, M)
    res = (gamma * M).sum()
    return res, gamma

def sinkhorn_stabilized(a, b, M, reg,
                             numItermax=1000, tau=1e3, stopThr=1e-9, 
                             warmstart=None, ilog=False):
    n = a.shape[0]
    m = b.shape[0]
    device = M.device
    dtype = M.dtype
    if warmstart is None:
        alpha = torch.zeros(n, dtype=dtype, device=device)
        beta = torch.zeros(m, dtype=dtype, device=device)
    else:
        alpha, beta = warmstart
    u = torch.ones(n, dtype=dtype, device=device) * (1/float(n))
    v = torch.ones(m, dtype=dtype, device=device) * (1/float(m))
    uprev = torch.empty(n, dtype=dtype, device=device)
    vprev = torch.empty(m, dtype=dtype, device=device)

    def get_K(alpha, beta):
        return torch.exp(-(M-alpha.view(-1,1)-beta.view(1,-1))/reg)
    def get_Gamma(alpha, beta, u, v):
        return torch.exp(-(M-alpha.view(-1,1)-beta.view(1,-1))/reg
                         + torch.log(u.view(-1,1)) + torch.log(v.view(1,-1)))

    K = get_K(alpha, beta)
    Kt = torch.transpose(K, 0, 1)
    loop = True
    err = 1.0
    numIter = 0
    while loop:
        uprev = u + 0.0
        vprev = v + 0.0
        v = b / (torch.mv(Kt, u) + 1e-16)
        u = a / (torch.mv(K,  v) + 1e-16)
        if torch.max(torch.abs(u)) > tau or torch.max(torch.abs(v)) > tau:
            alpha = alpha + reg * torch.log(u)
            beta = beta + reg * torch.log(v)
            u = torch.ones(n, dtype=dtype, device=device) * (1/n)
            v = torch.ones(m, dtype=dtype, device=device) * (1/m)
            K = get_K(alpha, beta)
            Kt = torch.transpose(K, 0, 1)
        if numIter % 10 == 0:
            transp = get_Gamma(alpha, beta, u, v)
            err = torch.norm(torch.sum(transp,0)-b) ** 2
        if err <= stopThr:
            loop = False
        if numIter >= numItermax:
            loop = False
        if torch.isnan(u).any() or torch.isnan(v).any():
            u = uprev + 0.0
            v = vprev + 0.0
            break
        numIter += 1
    if ilog:
        log = {}
        log['alpha'] = alpha + reg * torch.log(u)
        log['beta'] = beta + reg * torch.log(v)
    gamma = get_Gamma(alpha, beta, u, v)
    res = torch.sum(gamma * M)
    if ilog:
        return res, gamma, log
    else:
        return res, gamma

def sinkhorn_epsilon_scaling(a, b, M, reg,
                             numItermax=100, epsilon0=1e4, numInnerItermax=100,
                             tau=1e3, stopThr=1e-9):
    dtype = M.dtype
    device = M.device
    n = a.shape[0]
    m = b.shape[0]
    numItermin = 35
    numItermax = max(numItermin, numItermax)
    alpha = torch.zeros(n, dtype=dtype, device=device)
    beta = torch.zeros(m, dtype=dtype, device=device)
    def get_K(alpha, beta):
        return torch.exp(-(M-alpha.view(-1,1)-beta.view(1,-1))/reg)
    def get_reg(n):
        nn = torch.tensor(n, dtype=dtype)
        return (epsilon0 - reg) * torch.exp(-nn) + reg
    
    loop = True
    numIter = 0
    err = 1.0
    while loop:
        regi = get_reg(numIter)
        _, G, logi = sinkhorn_stabilized(a, b, M, regi, numItermax=numInnerItermax,
            stopThr=1e-9, warmstart=(alpha, beta), tau=tau, ilog=True)
        # print(regi)
        alpha = logi['alpha']
        beta = logi['beta']
        
        if numIter % 10 == 0:
            transp = G
            err = torch.norm(torch.sum(transp,0)-b)**2 \
                + torch.norm(torch.sum(transp,1)-a)**2
        # if err <= stopThr:
        #     loop = False
        if numIter >= numItermax:
            loop = False
        numIter += 1
    res = torch.sum(G * M)
    return res, G

def gaussian_diffusion(source_pts, locations, peaks, sigmas, nus, x_periodic=False):

    dtype = source_pts.dtype
    device = source_pts.device

    epsilon_cutoff = 1.0
    n = source_pts.size(0)
    m = locations.size(0)
    d = source_pts.size(1)
    x = source_pts.unsqueeze(1).expand(n, m, d)
    y = locations.unsqueeze(0).expand(n, m, d)
    Dx = torch.abs(x[:, :, 0] - y[:, :, 0])
    Dx = torch.where(Dx > 12.5, 25 - Dx, Dx)
    Dy = x[:, :, 1] - y[:, :, 1]
    D = torch.pow(Dx, 2) + torch.pow(Dy, 2)
    #D = torch.pow(x - y, 2)
    #D = D.sum(2)
    D_over_sigma = D / (sigmas.view(-1,1) ** 2)
    D_over_sigma_pow = torch.pow(D_over_sigma, nus.view(-1,1))
    P = torch.sum( peaks.view(-1,1) * torch.exp(-D_over_sigma_pow) / (np.sqrt(2*np.pi) * sigmas.view(-1, 1)), 0 )

    return P

'''
Script with all sampling methods. 

Computation of effective sampling size from:
https://github.com/jwalton3141/jwalton3141.github.io
following definition from:
ref Gelman, Andrew, J. B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin. 2013. Bayesian Data Analysis. Third Edition. London: Chapman & Hall / CRC Press.
'''

import numpy as np
import torch
from flonaco.phifour_utils import PhiFour


def run_langevin(model, x, n_steps, dt, beta):
    '''
    model: model with the potentiel we will run the langevin on
    x (tensor): init points for the chains to update (batch_dim, dim)
    dt -> is multiplied by N for the phiFour before being passed
    beta (float): inverse temperature
    '''
    optimizer = torch.optim.SGD([x], lr=dt)

    def add_noise(grad):
        noise = np.sqrt(2 / (dt * model.beta)) * torch.randn_like(grad)
        return grad + noise
    
    x.register_hook(add_noise)

    xs = []
    for t in range(n_steps):
        optimizer.zero_grad()
        loss = model.U(x).sum()
        loss.backward()
        optimizer.step()
        xs.append(x.clone())

    return torch.stack(xs)


def run_em_langevin(model, x, n_steps, dt, drift=None, bc=None, beta_ratios=None):
    '''
    Euler-Maruyama Scheme
    model (nn.Module): must implement potential energy function model.V(x)
    x (tensor): init points for the chains to update (batch_dim, dim)
    dt (float): integration timestep
    drift (tensor): non-conservative forcing term with dimensions of x
    bc (float): None or PBC (hypercube) specified as bc=L for [0,L]^d
    '''
    xs = []
    for t in range(n_steps):
        gradU = torch.autograd.grad(model.V(x), x)[0]
        x = x - (gradU - drift) * dt + np.sqrt(2 / model.beta * dt) * torch.randn_like(gradU)
        if bc is not None:
            x = x + bc * (x < 0.) - bc * (x > bc)
        xs.append(x.clone())
    return torch.stack(xs)


def run_action_langevin(model, xt, n_steps, dt_path, dt, bc=None):
    '''
    Path action langevin for diffusions
    model (nn.Module): model with the function `model.action(xt)` implemented
    xt (tensor): initial trajectory
    dt (float): integration timestep for the input trajectory
    drift (tensor): non-conservative forcing term with dimensions of x
    bc (float): None or PBC (hypercube) specified as bc=L for [0,L]^d
    '''
    xts = []
    for t in range(n_steps):
        gradPath = torch.autograd.grad(model.U(xt).sum(), xt)[0]
        noise = np.sqrt(2 * dt_path / (model.beta * model.dt)) * \
            torch.randn_like(gradPath)
        xt = xt - gradPath * dt_path + noise
        if bc is not None:
            xt = xt + bc * (xt < 0.) - bc * (xt > bc)
        xts.append(xt.clone())
    return torch.stack(xts)

def run_action_mh_langevin(model, target, xt, n_steps, dt_path, dt, bc=None):
    '''
    Path action langevin for diffusions
    model (nn.Module): model with the function `model.action(xt)` implemented
    xt (tensor): initial trajectory
    dt (float): integration timestep for the input trajectory
    drift (tensor): non-conservative forcing term with dimensions of x
    bc (float): None or PBC (hypercube) specified as bc=L for [0,L]^d
    '''
    xts = []
    accs = []
    betadt = target.dt * target.beta
    for t in range(n_steps):
        #with torch.no_grad():
        x = model.sample(xt.shape[0])
        xt = xt.reshape(-1,model.dim)

        ratio = -betadt * target.U(x) + betadt * model.nll(x.reshape(-1,model.dim))
        ratio += betadt * target.U(xt) - betadt * model.nll(xt.reshape(-1,model.dim))
        ratio = torch.exp(ratio)
        u = torch.rand_like(ratio)
        acc = u < torch.min(ratio, torch.ones_like(ratio))
        x[~acc] = xt[~acc]
        # xs.append(x.clone())
        accs.append(acc)
        xt.data = x.clone().detach()

        gradPath = torch.autograd.grad(target.U(xt).sum(), xt)[0]
        noise = np.sqrt(2 * dt_path / (betadt)) * \
            torch.randn_like(gradPath)
        xt = xt - gradPath * dt_path + noise
        if bc is not None:
            xt = xt +  bc * (xt < 0.) - bc * (xt > bc)
        xts.append(xt.clone())
    return torch.stack(xts), torch.stack(accs)



def run_metropolis(model, target, x_init, n_steps):
    xs = []
    accs = []

    for dt in range(n_steps):
        x = model.sample(x_init.shape[0])
        ratio = - target.beta * target.U(x) + model.nll(x)
        ratio += target.beta * target.U(x_init) - model.nll(x_init)
        ratio = torch.exp(ratio)
        u = torch.rand_like(ratio)
        acc = u < torch.min(ratio, torch.ones_like(ratio))
        x[~acc] = x_init[~acc]
        xs.append(x.clone())
        accs.append(acc)
        x_init = x.clone()

    return torch.stack(xs), torch.stack(accs)


def run_metrolangevin(model, target, x_lang, n_steps, dt, lag=1):
    '''
    model: model with the potential we will run the langevin on
    x (tensor): init points for the chains to update (batch_dim, dim)
    dt -> will be multiplied by N for the phiFour
    lag (int): number of Langevin steps before considering resampling
    '''
    optimizer = torch.optim.SGD([x_lang], lr=dt)

    def add_noise(grad):
        return grad + np.sqrt(2 / (dt * target.beta)) * torch.randn_like(grad)
    x_lang.register_hook(add_noise)

    xs = []
    accs = []
    for t in range(n_steps):
        if t % lag == 0:
            with torch.no_grad():
                x = model.sample(x_lang.shape[0])
                ratio = - target.beta * target.U(x) + model.nll(x)
                ratio += target.beta * target.U(x_lang) - model.nll(x_lang)
                ratio = torch.exp(ratio)
                u = torch.rand_like(ratio)
                acc = u < torch.min(ratio, torch.ones_like(ratio))
                x[~acc] = x_lang[~acc]
                accs.append(acc)
                x_lang.data = x.clone()

        optimizer.zero_grad()
        loss = target.U(x_lang).sum()
        loss.backward()
        optimizer.step()

        xs.append(x_lang.clone())

    return torch.stack(xs), torch.stack(accs)


def estimate_deltaF(model1, model2, 
                    xs_chains=None,
                    method='direct', 
                    no_ratio=False,
                    n_tot=int(1e4),
                    dt=1e-2,
                    state_dic=None,
                    coupling_per_site=False):
    """
    Estimate the ratio e^{-\beta (U_1(x) - U_2(x))} under the
    statistics of model2 with sampling defined by the sampling kwargs,
    or using the provided xs.

    If no_ratio -> estimates 1 under the statistics of model2 with sampling
    defined by the sampling kwargs (helped by model1 if mhlangevin or mh).

    Returns mean and variance (estimated according to sampling method)

    model1,2: RealNVP or MoG or PhiFour
    xs: samples to be used to compute the estimate - assumed to be from model2
    method: 'direct' computes the expectations using model2.sample
    method: 'mh' computes the expectations using run_metropolis
    method: 'mhlangevin' computes the expectations using run_metrolangevin
    state_dic: dictionary {'center':, 'width':, 'norm': }  or {'mean_thershold':, side: '+' or '-'}
    """
    if xs_chains is None:
        if method == 'direct':
            xs = model2.sample(n_tot)
            xs_chains = xs.unsqueeze(0)  # 1 iter of n_tot chains

        elif 'mh' in method:
            burn_in = int(1e2)
            steps_per_chain = int(1e2)
            assert n_tot / steps_per_chain >= 1
            x_init = model2.sample(int(n_tot / steps_per_chain))

            n_steps = steps_per_chain + burn_in
            x_init.detach_().requires_grad_()
            if method == 'mhlangevin':
                xs_all, _ = run_metrolangevin(model1, model2, x_init,
                                              n_steps, dt)
            elif method == 'mh':
                xs_all, _ = run_metropolis(model1, model2, x_init,
                                           n_steps=n_steps)

            xs = xs_all[-steps_per_chain:, :, :].reshape(-1, model1.dim)
            xs_chains = xs_all[-steps_per_chain:, :, :]
    else:
        print(RuntimeWarning('Error estimated according to method given'))
        xs = xs_chains.reshape(-1, model1.dim)
        n_tot = xs.shape[0]

    if no_ratio:
        Zs = torch.ones(n_tot, device=model2.device)
    else:
        Zs = Boltzmann_diff(xs, model1, model2,
                            coupling_per_site=coupling_per_site)
    
    if state_dic is not None:
        if 'width' in state_dic.keys():
            state_mask = (xs - state_dic['mean']
                          ).norm(state_dic['norm'], dim=1)
            state_mask = state_mask < state_dic['width']
        else:
            state_mask = xs.mean(-1) > state_dic['mean_threshold']
            state_mask = ~state_mask if state_dic['side'] == '-' else state_mask
        Zs = Zs * state_mask

    Zs_mean = Zs.mean().item()

    Zs_var = ((Zs ** 2).mean() - Zs_mean ** 2) 
    Zs_var = Zs_var.item()

    if method == 'direct':
        squared_weight = ((Zs ** 2).sum() / (Zs.sum() ** 2)).item()
        print('squared_weight: ', squared_weight, 'ratio Neff/N: ', 1 / (squared_weight * Zs.shape[0]))
        Zs_var = Zs_var * squared_weight 

    else:
        EES_per_dim = compute_ESS(xs_chains.detach().cpu())
        ESS = np.mean(compute_ESS(xs_chains.detach().cpu()))
        print('effective sample size per dim: ', EES_per_dim,
              '- ratio to total samples ',  ESS / Zs.shape[0])
        Zs_var = Zs_var / ESS 

    return Zs_mean, Zs_var, xs


def Boltzmann_diff(x, model1, model2, coupling_per_site=False):
    if coupling_per_site:
        diff_coupling = - model1.beta * model1.U_coupling_per_site(x)
        diff_coupling += model2.beta * model2.U_coupling_per_site(x)

        diff = diff_coupling.sum(dim=-1)
        diff += - model1.beta * model1.V(x) + model2.beta * model2.V(x)
    else:
        diff = - model1.beta * model1.U(x) + model2.beta * model2.U(x)

    return torch.exp(diff)


def compute_ESS(x):
    """
    Patching to take convention of axis orders,
    and convert from torch to numpy
    x : (n_iter, m_chaines, dim)
    """
    try:
        x = x.detach().numpy()
    except AttributeError:
        x = x

    x = x.swapaxes(0, 1)
    return my_ESS(x)


def my_ESS(x):
    """
    Compute the effective sample size of estimand of interest.
    Vectorised implementation.
    x : m_chaines, n_iter, dim
    """
    if x.shape < (2,):
        raise ValueError(
            'Calculation of effective sample size'
            'requires multiple chains of the same length.')
    try:
        m_chains, n_iter = x.shape
    except ValueError:
        return [my_ESS(y.T) for y in x.T]

    def variogram(t): return (
        (x[:, t:] - x[:, :(n_iter - t)])**2).sum() / (m_chains * (n_iter - t))

    post_var = my_gelman_rubin(x)
    assert post_var > 0

    t = 1
    rho = np.ones(n_iter)
    negative_autocorr = False

    # Iterate until the sum of consecutive estimates of autocorrelation is negative
    while not negative_autocorr and (t < n_iter):
        rho[t] = 1 - variogram(t) / (2 * post_var)

        if not t % 2:
            negative_autocorr = sum(rho[t - 1:t + 1]) < 0

        t += 1

    return int(m_chains * n_iter / (1 + 2 * rho[1:t].sum()))


def my_gelman_rubin(x):
    """
    Estimate the marginal posterior variance. Vectorised implementation.
    x : m_chaines, n_iter
    """
    m_chains, n_iter = x.shape

    # Calculate between-chain variance
    B_over_n = ((np.mean(x, axis=1) - np.mean(x))**2).sum() / (m_chains - 1)

    # Calculate within-chain variances
    W = ((x - x.mean(axis=1, keepdims=True)) **
         2).sum() / (m_chains * (n_iter - 1))

    # (over) estimate of variance
    s2 = W * (n_iter - 1) / n_iter + B_over_n

    return s2

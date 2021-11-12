'''
Simplified implementation of Real-NVPs borrowing from
https://github.com/chrischute/real-nvp.

Original paper:
Density estimation using Real NVP
Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio
arXiv:1605.08803
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flonaco.models import MLP
from torch.distributions.multivariate_normal import MultivariateNormal
from flonaco.two_channel import bridge_energy, get_bridge


class ResidualAffineCoupling(nn.Module):
    """ Residual Affine Coupling layer 
    Implements coupling layers with a rescaling 
    Args:
        s (nn.Module): scale network
        t (nn.Module): translation network
        mask (binary tensor): binary array of same 
        dt (float): rescaling factor for s and t
    """

    def __init__(self, s=None, t=None, mask=None, dt=1):
        super(ResidualAffineCoupling, self).__init__()

        self.mask = mask
        self.scale_net = s
        self.trans_net = t
        self.dt = dt

    def forward(self, x, log_det_jac=None, inverse=False):
        if log_det_jac is None:
            log_det_jac = 0

        s = self.mask * self.scale_net(x * (1 - self.mask))
        s = torch.tanh(s)
        t = self.mask * self.trans_net(x * (1 - self.mask))

        s = self.dt * s
        t = self.dt * t

        if inverse:
            if torch.isnan(torch.exp(-s)).any():
                raise RuntimeError('Scale factor has NaN entries')
            log_det_jac -= s.view(s.size(0), -1).sum(-1)

            x = x * torch.exp(-s) - t

        else:
            log_det_jac += s.view(s.size(0), -1).sum(-1)
            x = (x + t) * torch.exp(s)
            if torch.isnan(torch.exp(s)).any():
                raise RuntimeError('Scale factor has NaN entries')

        return x, log_det_jac


class RealNVP_MLP(nn.Module):
    """ Minimal Real NVP architecture

    Args:
        dims (int,): input dimension
        n_realnvp_blocks (int): number of pairs of coupling layers
        block_depth (int): repetition of blocks with shared param
        init_weight_scale (float): scaling factor for weights in s and t layers
        prior_arg (dict): specifies the base distribution
        mask_type (str): 'half' or 'inter' masking pattern
        hidden_dim (int): # of hidden neurones per layer (coupling MLPs)
    """

    def __init__(self, dim, n_realnvp_blocks, 
                 block_depth,
                 init_weight_scale=None,
                 prior_arg={'type': 'standn'},
                 mask_type='half',  
                 hidden_dim=10,
                 hidden_depth=3,
                 hidden_bias=True,
                 hidden_activation=torch.relu,
                 device='cpu'):
        super(RealNVP_MLP, self).__init__()

        self.device = device
        self.dim = dim
        self.n_blocks = n_realnvp_blocks
        self.block_depth = block_depth
        self.couplings_per_block = 2  # one update of entire layer per block 
        self.n_layers_in_coupling = hidden_depth  # depth of MLPs in coupling layers 
        self.hidden_dim_in_coupling = hidden_dim
        self.hidden_bias = hidden_bias
        self.hidden_activation = hidden_activation
        self.init_scale_in_coupling = init_weight_scale

        mask = torch.ones(dim, device=self.device)
        if mask_type == 'half':
            mask[:int(dim / 2)] = 0
        elif mask_type == 'inter':
            idx = torch.arange(dim, device=self.device)
            mask = mask * (idx % 2 == 0)
        else:
            raise RuntimeError('Mask type is either half or inter')
        self.mask = mask.view(1, dim)

        self.coupling_layers = self.initialize()

        self.beta = 1.  # effective temperature needed e.g. in Langevin

        self.prior_arg = prior_arg

        if prior_arg['type'] == 'standn':
            self.prior_prec =  torch.eye(dim).to(device)
            self.prior_log_det = 0
            self.prior_distrib = MultivariateNormal(
                torch.zeros((dim,), device=self.device), self.prior_prec)

        elif prior_arg['type'] == 'uncoupled':
            self.prior_prec = prior_arg['a'] * torch.eye(dim).to(device)
            self.prior_log_det = - torch.logdet(self.prior_prec)
            self.prior_distrib = MultivariateNormal(
                torch.zeros((dim,), device=self.device),
                precision_matrix=self.prior_prec)

        elif prior_arg['type'] == 'coupled':
            self.beta_prior = prior_arg['beta']
            self.coef = prior_arg['alpha'] * dim
            prec = torch.eye(dim) * (3 * self.coef + 1 / self.coef)
            prec -= self.coef * torch.triu(torch.triu(torch.ones_like(prec),
                                                      diagonal=-1).T, diagonal=-1)
            prec = prior_arg['beta'] * prec
            self.prior_prec = prec.to(self.device)
            self.prior_log_det = - torch.logdet(prec)
            self.prior_distrib = MultivariateNormal(
                torch.zeros((dim,), device=self.device),
                precision_matrix=self.prior_prec)

        elif prior_arg['type'] == 'white':
            cov = prior_arg['cov']
            self.prior_prec = torch.inverse(cov).to(device)
            self.prior_prec = 0.5 * (self.prior_prec + self.prior_prec.T)
            self.prior_mean = prior_arg['mean'].to(device)
            self.prior_log_det = - torch.logdet(self.prior_prec)
            self.prior_distrib = MultivariateNormal(
                prior_arg['mean'],
                precision_matrix=self.prior_prec
                )

        elif prior_arg['type'] == 'bridge':
            self.bridge_kwargs = prior_arg['bridge_kwargs']

        else:
            raise NotImplementedError("Invalid prior arg type")

    def forward(self, x, return_per_block=False):
        log_det_jac = torch.zeros(x.shape[0], device=self.device)

        if return_per_block:
            xs = [x]
            log_det_jacs = [log_det_jac]

        for block in range(self.n_blocks):
            couplings = self.coupling_layers[block]

            for dt in range(self.block_depth):
                for coupling_layer in couplings:
                    x, log_det_jac = coupling_layer(x, log_det_jac)

                if return_per_block:
                    xs.append(x)
                    log_det_jacs.append(log_det_jac)

        if return_per_block:
            return xs, log_det_jacs
        else:
            return x, log_det_jac

    def backward(self, x, return_per_block=False):
        log_det_jac = torch.zeros(x.shape[0], device=self.device)

        if return_per_block:
            xs = [x]
            log_det_jacs = [log_det_jac]
        
        for block in range(self.n_blocks):
            couplings = self.coupling_layers[::-1][block]

            for dt in range(self.block_depth):
                for coupling_layer in couplings[::-1]:
                    x, log_det_jac = coupling_layer(
                        x, log_det_jac, inverse=True)

                if return_per_block:
                    xs.append(x)
                    log_det_jacs.append(log_det_jac)

        if return_per_block:
            return xs, log_det_jacs
        else:
            return x, log_det_jac

    def initialize(self):
        dim = self.dim
        coupling_layers = []

        for block in range(self.n_blocks):
            layer_dims = [self.hidden_dim_in_coupling] * \
                (self.n_layers_in_coupling - 2)
            layer_dims = [dim] + layer_dims + [dim]

            couplings = self.build_coupling_block(layer_dims)

            coupling_layers.append(nn.ModuleList(couplings))

        return nn.ModuleList(coupling_layers)

    def build_coupling_block(self, layer_dims=None, nets=None, reverse=False):
        count = 0
        coupling_layers = []
        for count in range(self.couplings_per_block):
            s = MLP(layer_dims, init_scale=self.init_scale_in_coupling)
            s = s.to(self.device)
            t = MLP(layer_dims, init_scale=self.init_scale_in_coupling)
            t = t.to(self.device)

            if count % 2 == 0:
                mask = 1 - self.mask
            else:
                mask = self.mask
            
            dt = self.n_blocks * self.couplings_per_block * self.block_depth
            dt = 2 / dt
            coupling_layers.append(ResidualAffineCoupling(
                s, t, mask, dt=dt))

        return coupling_layers

    def nll(self, x):
        z, log_det_jac = self.backward(x)

        if self.prior_arg['type']=='bridge':
            a_min = torch.tensor([self.bridge_kwargs["x0"],self.bridge_kwargs["y0"]])
            b_min = torch.tensor([self.bridge_kwargs["x1"],self.bridge_kwargs["y1"]])
            dt = self.bridge_kwargs["dt"]
            prior_nll = bridge_energy(z, dt=dt, a_min=a_min, b_min=b_min, device=self.device)
            return prior_nll - log_det_jac
        elif self.prior_arg['type'] == 'white':
                z = z - self.prior_mean

        prior_ll = - 0.5 * torch.einsum('ki,ij,kj->k', z, self.prior_prec, z)
        prior_ll -= 0.5 * (self.dim * np.log(2 * np.pi) + self.prior_log_det)

        ll = prior_ll + log_det_jac
        nll = -ll
        return nll

    def sample(self, n):
        if self.prior_arg['type'] == 'standn':
            z = torch.randn(n, self.dim, device=self.device)
        elif self.prior_arg['type'] == 'bridge':
            # get a bridge
            n_steps = self.bridge_kwargs["n_steps"]
            bridges = torch.zeros(n, n_steps - 2, 1, 2, device=self.device)
            for i in range(n):
                bridges[i,:] = get_bridge(**self.bridge_kwargs)
            z = bridges.detach().requires_grad_().view(n, -1)
        else:
            z = self.prior_distrib.rsample(torch.Size([n, ])).to(self.device)

        return self.forward(z)[0]

    def U(self, x):
        """
        alias
        """
        return self.nll(x)



    def V(self, x):
        z, log_det_jac = self.backward(x)
        return self.beta_prior * (z ** 2 / 2).sum(dim=-1) / self.coef - log_det_jac

    def U_coupling_per_site(self, x):
        """
        return the (\nable phi) ** 2 to be used in direct computation
        with dirichlet boundary conditions to 0
        U = U_coup_per_site.sum(dim=-1) + V
        """
        bc_value = 0
        z, _ = self.backward(x)

        z_ = F.pad(input=z, pad=(1,) * 2, mode='constant',
                   value=bc_value)

        return ((z_[:, 1:] - z_[:, :-1]) ** 2 / 2) * self.coef * self.beta_prior

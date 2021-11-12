import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.multivariate_normal import MultivariateNormal

DIM_PHYS = 1  # Implementation for 1d on physical system 

class PhiFour(nn.Module):
    def __init__(self, a, b, dim_grid, dim_phys=DIM_PHYS,
                 beta=1,
                 bc=('dirichlet', 0),
                 tilt=None,
                 dtype=torch.float32, 
                 device='cpu'):
        """
        Class to handle operations around Stochastic Allen-Cahn model
        Args:
            a: coupling term coef
            b: local field coef
            dim_grid: grid size along one physical dimension
            dim_phys: number of dimensions of the physical grid
            beta: inverse temperature
            tilt: None or {"val":0.7, "lambda":1} - mean value + Lagrange param
        """
        self.device = device

        self.a = a
        self.b = b
        self.beta = beta
        self.dim_grid = dim_grid
        self.dim_phys = dim_phys
        self.sum_dims = tuple(i + 1 for i in range(dim_phys))

        self.bc = bc
        self.tilt = tilt

    def init_field(self, n_or_values):
        if isinstance(n_or_values, int):
            x = torch.rand((n_or_values,) + (self.dim_grid,) * self.dim_phys)
            x = x * 2 - 1
        else:
            x = n_or_values
        return x

    def V(self, x):
        coef = self.a * self.dim_grid
        V = ((1 - x ** 2) ** 2 / 4 + self.b * x).sum(self.sum_dims) / coef
        if self.tilt is not None: 
            tilt = (self.tilt['val'] - x.mean(self.sum_dims)) ** 2 
            tilt = self.tilt["lambda"] * tilt / (4 * self.dim_grid)
            V += tilt
        return V

    def U(self, x):
        # Does not include the temperature! need to be explicitely added in Gibbs factor

        if self.bc[0] == 'dirichlet':
            x_ = F.pad(input=x, pad=(1,) * (2*self.dim_phys), mode='constant',
                      value=self.bc[1])
        else:
            raise NotImplementedError("Only dirichlet BC implemeted")

        grad_term = ((x_[:, 1:, ...] - x_[:, :-1, ...]) ** 2 / 2).sum(self.sum_dims)
        if self.dim_phys == 2:
            grad_term += ((x_[:, :, 1:] - x_[:, :, :-1]) ** 2 / 2).sum(self.sum_dims)
        
        coef = self.a * self.dim_grid
        return grad_term * coef + self.V(x) 

    def U_coupling_per_site(self, x):
        """
        return the (\nabla phi) ** 2 to be used in direct computation
        """
        assert self.dim_phys == 1
        if self.bc[0] == 'dirichlet':
            x_ = F.pad(input=x, pad=(1,) * (2*self.dim_phys), mode='constant',
                      value=self.bc[1])
        else:
            raise NotImplementedError("Only dirichlet BC implemeted")

        return ((x_[:, 1:] - x_[:, :-1]) ** 2 / 2) * self.a * self.dim_grid

    def grad_U(self, x_init):
        x = x_init.detach()
        x = x.requires_grad_()
        optimizer = torch.optim.SGD([x], lr=0)
        optimizer.zero_grad()
        loss = self.U(x).sum()
        loss.backward()
        return x.grad.data

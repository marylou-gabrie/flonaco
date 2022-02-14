from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch._C import device
import torch.distributions as td
import torch.nn as nn

class Croissants(nn.Module):
    def __init__(self, means, covars, 
                ring_mean,
                ring_var,
                weights=None,
                wiggle=False,
                dtype=torch.float32, device='cpu'):
        """
        Class to handle operations around mixtures of multivariate 
        Gaussian distributions
        Args:
            means: list of 1d tensors of centroids
            covars: list of 2d tensors of covariances
            ring_mean: float, mean of ring
            ring_var: float, width of the ring
            weights: 
        """
        self.device = device
        self.beta = 1. # for sampling with langevin and mh
        self.means = means
        self.covars = covars
        self.ring_mean = ring_mean
        self.ring_var = ring_var
        self.dim = means[0].shape[0]
        self.k = len(means) # number of components in the mixture
        self.wiggle = wiggle

        if weights is not None:
            self.weights = torch.tensor(weights, dtype=dtype, device=device)
        else:
            self.weights = torch.tensor([1 / self.k] * self.k,
                                        dtype=dtype, device=device)
        
        self.covars_inv = torch.stack([torch.inverse(cv) for cv in covars])
        self.dets =  torch.stack([torch.det(cv) for cv in covars])

    def U(self, x):
        x = x.unsqueeze(1)
        m = torch.stack(self.means).unsqueeze(0)
        if self.wiggle:
            centered_x = x[:, :, :1] - m 
            centered_x -= torch.sin(5 * x[:, :, 1:] / self.ring_mean).to(self.device)
        else:
            centered_x = x - m
        args = - 0.5 * torch.einsum('kci,cij,kcj->kc', centered_x,
                                     self.covars_inv, centered_x)
        args += torch.log(self.weights)
        args -= torch.log((self.weights.sum() * torch.sqrt((2 * np.pi) ** self.dim * self.dets)))

        mog_U = - torch.logsumexp(args, 1)
        x = x.squeeze(1)
        ring_U = 0.5 * (torch.norm(x, p=2, dim=1) - self.ring_mean) ** 2 
        ring_U /= self.ring_var
        
        return mog_U + ring_U
    
    def grad_U(self, x_init):
        x = x_init.detach()
        x = x.requires_grad_()
        optimizer = torch.optim.SGD([x], lr=0)
        optimizer.zero_grad()
        loss = self.U(x).sum()
        loss.backward()
        return x.grad.data


def plot_2d_level(model, x_min=-10, x_max=10,
                    y_min=None, y_max=None,
                    n_points=100, ax=None, title=''):
    """
    Args:
    model (RealNVP_MLP or MoG): must have a .sample and .U method
    """
    # if 
    x_range = torch.linspace(x_min, x_max, n_points, device=model.device)
    if y_min is None:
        y_range = x_range.clone()
    else:
        y_range = torch.linspace(y_min, y_max, n_points, device=model.device)

    grid = torch.meshgrid(x_range, y_range)
    xys = torch.stack(grid).reshape(2, n_points ** 2).T.to(model.device)
    if model.dim > 2:
        blu = torch.zeros(n_points ** 2, model.dim).to(model.device)
        blu[:, 0:2] = xys
        xys = blu

    Us = model.U(xys).reshape(n_points, n_points).T.detach().cpu().numpy()
    
    if ax is None:
        plt.figure()
    else:
        plt.sca(ax)
    plt.imshow(np.exp(- Us[::-1])
        , cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.title(title)
    return Us

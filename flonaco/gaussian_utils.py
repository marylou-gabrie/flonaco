from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn

class MoG(nn.Module):
    def __init__(self, means, covars, weights=None,
                 dtype=torch.float32, device='cpu'):
        """
        Class to handle operations around mixtures of multivariate
        Gaussian distributions
        Args:
            means: list of 1d tensors of centroids
            covars: list of 2d tensors of covariances
            weights: list of relative statistical weights (does not need to sum to 1)
        """
        self.device = device
        self.beta = 1.  # model 'temperature' for sampling with langevin and mh
        self.means = means
        self.covars = covars
        self.dim = means[0].shape[0]
        self.k = len(means)  # number of components in the mixture

        if weights is not None:
            self.weights = torch.tensor(weights, dtype=dtype, device=device)
        else:
            self.weights = torch.tensor([1 / self.k] * self.k,
                                        dtype=dtype, device=device)

        self.cs_distrib = td.categorical.Categorical(probs=self.weights)
        self.normal_distribs = []
        for c in range(self.k):
            c_distrib = td.multivariate_normal.MultivariateNormal(
                self.means[c].to(device),
                covariance_matrix=self.covars[c].to(device)
                )
            self.normal_distribs.append(c_distrib)

        self.covars_inv = torch.stack([torch.inverse(cv) for cv in covars])
        self.dets = torch.stack([torch.det(cv) for cv in covars])

    def sample(self, n):
        cs = self.cs_distrib.sample_n(n).to(self.device)

        samples = torch.zeros((n, self.dim), device=self.device)
        for c in range(self.k):
            n_c = (cs == c).sum()
            samples[cs == c, :] = self.normal_distribs[c].sample_n(n_c)
        return samples.to(self.device)

    def U(self, x):
        x = x.unsqueeze(1)
        m = torch.stack(self.means).unsqueeze(0)
        args = - 0.5 * torch.einsum('kci,cij,kcj->kc', x-m, self.covars_inv, x-m)
        args += torch.log(self.weights)
        args -= torch.log((self.weights.sum() * torch.sqrt((2 * np.pi) ** self.dim * self.dets)))
        return - torch.logsumexp(args, 1)


def plot_2d_level(model, x_min=-10, x_max=10,
                    y_min=None, y_max=None,
                    n_points=100, ax=None, title=''):
    """
    Plot "push-forward" of base density.

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
    Us = model.U(xys).reshape(n_points, n_points).T.detach().cpu().numpy()
    
    if ax is None:
        plt.figure()
    else:
        plt.sca(ax)
    plt.imshow(np.exp(- Us[::-1]))
    plt.axis('off')
    plt.colorbar()
    plt.title(title)
    return Us

def plot_2d_level_reversed(model, target, x_min=-10, x_max=10,
                    n_points=100, ax=None, title=''):
    """
    Plot "push-backward" of the target density 

    Args:
    model (RealNVP_MLP)
    target (MoG): need to have a U method
    """
    x_range = torch.linspace(x_min, x_max, n_points, device=model.device)
    y_range = x_range.clone()

    grid = torch.meshgrid(x_range, y_range)
    xys = torch.stack(grid).reshape(2, n_points ** 2).T.to(model.device)
    Gxys, logdetjacs = model.forward(xys) 
    Us = (target.U(Gxys) - logdetjacs).reshape(n_points, n_points).T.detach().cpu().numpy()
    
    if ax is None:
        plt.figure()
    else:
        plt.sca(ax)
    plt.imshow(np.exp(- Us[::-1]))
    plt.axis('off')
    plt.colorbar()
    plt.title(title)
    return Us



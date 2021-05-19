import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from flonaco.real_nvp_mlp import RealNVP_MLP

def plot_map_point_cloud(model, n_points=100):
    """
    model (RealNVP): must have a .forward method and .prior_distrib attribute
    """
    z = model.prior_distrib.rsample(torch.Size([n_points,]))
    x, _ = model.forward(z)

    z = z.detach()
    x = x.detach()

    plt.figure(figsize=(15,5 * int(model.dim / 10)))
    axs = [plt.subplot(math.ceil(model.dim/5), 5, i + 1) for i in range(model.dim)]
    for i in range(model.dim):
        plt.sca(axs[i])
        plt.plot(x[:, i], z[:,i],'o', ms=1.0)
        plt.xlabel(r'$z_{:d}$'.format(i +1))
        plt.ylabel(r'$G(z)_{:d}$'.format(i +1))
    plt.tight_layout()

def plot_map_point_cloud_Fourier(model, n_points=100):
    """
    model (RealNVP): must have a .forward method and .prior_distrib attribute
    """
    z = model.prior_distrib.rsample(torch.Size([n_points,]))
    x, _ = model.forward(z)

    z_i = torch.zeros(n_points, model.dim, 2)
    z_i[:, :, 0] = z
    z = torch.fft(z_i, signal_ndim=1)

    x_i = torch.zeros(n_points, model.dim, 2)
    x_i[:, :, 0] = x
    x = torch.fft(x_i, signal_ndim=1)

    z = z.detach()
    x = x.detach()

    plt.figure(figsize=(15, 2.5 * int(model.dim / 10)))
    axs = [plt.subplot(math.ceil(model.dim/10) + 1, 5, i + 1) for i in range(int(model.dim/2) + 1)]
    for i in range(int(model.dim / 2 + 1)):
        plt.sca(axs[i])
        plt.plot(z[:,i, 1], x[:, i, 1], 'o', ms=1.0, label='imaginary',alpha=0.5)
        plt.plot(z[:,i, 0], x[:, i, 0], 'o', ms=1.0, label='real',alpha=0.5)
        plt.xlabel(r'$F(Z)_{:d}$'.format(i +1))
        plt.ylabel(r'$F(G(Z))_{:d}$'.format(i +1))
    plt.legend()
    plt.tight_layout()

def plot_Fourier_spectrum(model_or_chains, n_points=100):
    """
    model (RealNVP): must have a .forward method and .prior_distrib attribute
    or x: tensor of shape (npoints, dim)
    """
    if isinstance(model_or_chains, RealNVP_MLP):
        model = model_or_chains
        z = model.prior_distrib.rsample(torch.Size([n_points,]))
        x, _ = model.forward(z)
    else:
        x = model_or_chains.detach()
        n_points = x.shape[0]

    dim = x.shape[-1]
    x_i = torch.zeros(n_points, dim, 2)
    x_i[:, :, 0] = x
    x = torch.fft(x_i, signal_ndim=1)

    x = x.detach()

    plt.figure(figsize=(15, 5))
    
    axs = [plt.subplot(131), plt.subplot(132), plt.subplot(133)]
    print(x.shape)
    kmax = int(dim / 2 + 1)
    for i in range(n_points):
        plt.sca(axs[0])
        plt.plot(x[i, :kmax, 0], 'o-', ms=1.0, label='real', c='C0', alpha=0.25)
        plt.title('real')
        plt.xlabel(r'$k$')
        plt.ylabel(r'Re$(F(G(Z))_k)$')

        plt.sca(axs[1])
        plt.plot(x[i, :kmax, 1], 'o-', ms=1.0, label='imaginary', c='C1', alpha=0.25)
        plt.xlabel(r'$k$')
        plt.ylabel(r'Im$(F(G(Z))_k)$')
        plt.title('imaginary')

        plt.sca(axs[2])
        plt.plot(x[i, :kmax, 0] ** 2 + x[i, :kmax, 1] ** 2, 
                 'o-', ms=1.0, label='real', c='C2', alpha=0.25)
        plt.title('power spectrum')
        plt.xlabel(r'$k$')
        plt.ylabel(r'$\vert F(G(Z))_k \vert^2$')
        plt.yscale('log')
        plt.xscale('log')

    plt.sca(axs[0])
    plt.plot(x[:, :kmax, 0].mean(dim=0), 'o-', ms=1.0, label='real', c='k')

    plt.sca(axs[1])
    plt.plot(x[:, :kmax, 1].mean(dim=0), 'o-', ms=1.0, label='imaginary', c='k')
   
    plt.sca(axs[2])
    plt.plot((x[:, :kmax, 0] ** 2 + x[:, :kmax, 1] ** 2).mean(dim=0),
                'o-', ms=1.0, label='real', c='k')
    plt.plot(np.arange(1, kmax, 1), 1/np.arange(1, kmax, 1)**2, 
                label='$1/k^2$')

    plt.tight_layout()
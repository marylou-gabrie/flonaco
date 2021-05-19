import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import copy


class TwoChannel(nn.Module):
    def __init__(self, a_center=None, b_center=None,
                 bc=('dirichlet', 0),
                 dtype=torch.float32, device='cpu'):
        """
        Implementation of a random rugged potential with a non-conservative force
        Args:
            n_hills: number of hills in the potential
            i_range: range of integers for the hills
            sinv: width parameter for the metastable state
        """

        self.device = device

        self.dim = 2
        self.beta = 1.
        self.dt = 1.5e-4
        self.n_steps = 100
        #self.drift = torch.zeros(1, self.dim, device=device)


        # set up the potential
        self.mu1 = torch.tensor([[0., 1./3.]], device=device)
        self.mu2 = torch.tensor([[0., 5./3.]], device=device)
        self.mu3 = torch.tensor([[1., 0.]], device=device)
        self.mu4 = torch.tensor([[-1., 0.]], device=device)

        self.A1 = 30.
        self.A2 = -30.
        self.A3 = -50.
        self.A4 = -50.


        # set the endpoints
        self.a_center = self.mu4.clone()
        self.b_center = self.mu3.clone()

        # minimizer for GD initiated at a_center, b_center
        self.a_min = self.mu4.clone()
        self.b_min = self.mu3.clone()

    def locate_minima(self):
        def minimize(x, dt=1e-5, tol=1e-4):
            grad = torch.autograd.grad(self.V(x), x)[0]
            grad_norm = (grad**2).sum()
            while grad_norm > tol:
                x = x - grad * dt
                grad = torch.autograd.grad(self.V(x), x)[0]
                grad_norm = (grad**2).sum()
            return x

        x_init = self.a_center.clone().requires_grad_()
        xa = minimize(x_init).detach()
        x_init = self.b_center.clone().requires_grad_()
        xb = minimize(x_init).detach()
        self.a_min = xa
        self.b_min = xb

    @staticmethod
    def g(x, mu, amp):
        """ just a Gaussian """
        return amp * torch.exp(-torch.sum((x - mu)**2,dim=-1))

    def V(self, x):
        """ Metzner ref """
        #x = x.reshape(-1,2)
        mixture = self.g(x, self.mu1, self.A1) + self.g(x, self.mu2, self.A2) +\
        self.g(x, self.mu3, self.A3) + self.g(x, self.mu4, self.A4) + 0.2 * torch.sum((x - self.mu1)**4, dim=-1)
        return mixture

    def drift(self, xt):
        return torch.zeros(xt.shape, device=self.device).reshape(xt.shape[0],-1,1,2)
    
    def U(self, xt):
        xt = xt.view(xt.shape[0],-1, 1, 2)
        grads = torch.autograd.grad(self.V(xt).sum(), xt, create_graph=True)[0]
        drift = self.drift(xt)
        # x_{i+1} - x_i / dt + \nabla V(x_i)
        integrand = (xt[:,1:,:,:] - xt[:,:-1,:,:]) / self.dt + grads[:,:-1,:,:] - drift[:,:-1,:,:]
        bc_a = torch.sum(
            ((xt[:,0,:,:] - self.a_min) / self.dt + grads[:,0,:,:] - drift[:,0,:,:])**2, dim=(1,2))
        bc_b = torch.sum(
            ((self.b_min - xt[:,-1,:,:]) / self.dt + grads[:,-1,:,:] - drift[:,-1,:,:])**2, dim=(1,2))
        return self.dt*(torch.sum(integrand**2,dim=(1,2,3)) + bc_a + bc_b) / 4.0

    def plot(self, filled=True, levels=20, figax=None, filename=None, colorbar=True):

        from matplotlib.colors import LinearSegmentedColormap
        stanford = LinearSegmentedColormap.from_list(
            "stanford", ['#006B81', '#f1f8f1', '#8C1515'])

        matplotlib.cm.register_cmap(name="stanford", cmap=stanford)
        #plt.style.use("rotskoff")

        xs = torch.linspace(-2., 2., 100, device=self.device)
        ys = torch.linspace(-2., 2., 100, device=self.device)
        Xs, Ys = torch.meshgrid(xs, ys)

        with torch.no_grad():
            Zs = torch.zeros(Xs.shape)
            for i in range(len(Xs)):
                for j in range(len(Xs[i])):
                    Zs[i, j] = self.V(torch.tensor(
                        [Xs[i, j], Ys[i, j]], dtype=torch.float,device=self.device))

        plot_data = Xs.cpu().numpy(), Ys.cpu().numpy(), Zs.cpu().numpy()

        if figax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig, ax = figax
        ax.set_ylim(-2., 2.)
        ax.set_xlim(-2., 2.)
        if filled:
            plot = ax.contourf(*plot_data, cmap="stanford", levels=levels)
        else:
            plot = ax.contour(*plot_data, cmap="stanford", levels=levels)
        fig.tight_layout()
        if colorbar:
            fig.colorbar(plot)

        if filename is not None:
            fig.savefig(filename)

        return fig, ax, plot

    def plot3d(self, figax=None, filename=None):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.colors import LinearSegmentedColormap
        stanford = LinearSegmentedColormap.from_list(
            "stanford", ['#006B81', '#f1f8f1', '#8C1515'])

        matplotlib.cm.register_cmap(name="stanford", cmap=stanford)
        plt.style.use("rotskoff")

        xs = torch.linspace(-2., 2., 100, device=self.device)
        ys = torch.linspace(-2., 2., 100, device=self.device)
        Xs, Ys = torch.meshgrid(xs, ys)

        with torch.no_grad():
            Zs = torch.zeros(Xs.shape)
            for i in range(len(Xs)):
                for j in range(len(Xs[i])):
                    Zs[i, j] = self.V(torch.tensor(
                        [Xs[i, j], Ys[i, j]], dtype=torch.float, device=self.device))

        plot_data = Xs.cpu().numpy(), Ys.cpu().numpy(), Zs.cpu().numpy()

        if figax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = figax
        ax.set_ylim(-2., 2.)
        ax.set_xlim(-2., 2.)

        plot = ax.plot_surface(*plot_data, cmap="stanford")
        fig.tight_layout()

        if filename is not None:
            fig.savefig(filename)

        return fig, ax, plot

    def get_bridge_args(self):
        return dict(n_steps=self.n_steps,
                    beta=self.beta,
                    x0=self.a_min[0, 0],
                    x1=self.b_min[0, 0],
                    y0=self.a_min[0, 1],
                    y1=self.b_min[0, 1],
                    t0=0.,
                    t1=self.dt * self.n_steps,
                    dt=self.dt,
                    device=self.device,
                    )


def get_bridge(n_steps, beta=1.0, x0=0.25, x1=0.75, y0=0.25, y1=0.75, t0=0., t1=0.015, dt=1e-2, device="cpu"):
    steps_x = torch.randn(n_steps, device=device)
    steps_y = torch.randn(n_steps, device=device)
    steps_x[0] = 0.
    steps_y[0] = 0.
    # generate the Brownian process
    ts = torch.linspace(t0, t1, n_steps, device=device)
    wT_x = torch.cumsum(steps_x, 0) * np.sqrt((t1 - t0) /
                                              n_steps) * np.sqrt(2. / beta)
    wT_y = torch.cumsum(steps_y, 0) * np.sqrt((t1 - t0) /
                                              n_steps) * np.sqrt(2. / beta)
    # condition and shift means
    path_x = wT_x - wT_x[-1] * (ts - t0) / (t1 - t0) + \
        x0 + (ts - t0) * (x1 - x0) / (t1 - t0)
    path_y = wT_y - wT_y[-1] * (ts - t0) / (t1 - t0) + \
        y0 + (ts - t0) * (y1 - y0) / (t1 - t0)

    bridge = torch.zeros(n_steps - 2, 2, device=device)
    bridge[:, 0] = path_x[1:-1]
    bridge[:, 1] = path_y[1:-1]
    bridge = bridge.requires_grad_().reshape(n_steps - 2, 1, 2)

    return bridge


def bridge_energy(bridge, dt=1e-2, a_min=torch.tensor([0.25, 0.25]), b_min=torch.tensor([0.75, 0.75]), device="cpu"):
    a_min = a_min.to(device)
    b_min = b_min.to(device)
    n_bridges = bridge.shape[0]
    bridge = bridge.view(n_bridges, -1, 1, 2)
    integrand = (bridge[:, 1:, :, :] - bridge[:, :-1, :, :]) / dt
    bc_a = torch.sum(((bridge[:, 0, :, :] - a_min) / dt)**2, dim=(1, 2))
    bc_b = torch.sum(((b_min - bridge[:, -1, :, :]) / dt)**2, dim=(1, 2))
    return dt*(torch.sum(integrand**2, dim=(1, 2, 3)) + bc_a + bc_b) / 4.0

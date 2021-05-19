import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import argparse
from realnvp.real_nvp_mlp import RealNVP_MLP

import os
import time
import pickle
import torch.nn.functional as F

from realnvp.real_nvp_mlp import RealNVP_MLP
from realnvp.sampling import estimate_deltaF
from realnvp.training import train
from realnvp.utils_io import get_file_name

# parse args
parser = argparse.ArgumentParser(description='Two c')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--n-langevin-steps', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('--n-bridges', type=int, default=20, metavar='N',
                    help='number of walkers (default: 10)')
parser.add_argument('--dt', type=float, default=4e-3, metavar='N',
                    help='bridge discretization time (default: 4e-3)')
parser.add_argument('--dtp', type=float, default=1e-5, metavar='N',
                    help='bridge discretization time (default: 4e-3)')
parser.add_argument('--bridge-len', type=int, default=50, metavar='N',
                    help='number of walkers (default: 50)')
parser.add_argument('--beta', type=float, default=2., metavar='N',
                    help='bridge temp (default: 2)')
parser.add_argument('--js', type=bool, default=False, metavar='N',
                    help='train with symmetrized loss')
parser.add_argument('--noneq', type=bool, default=False, metavar='N',
                    help='train with nonequilibrium force')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=78123989, metavar='S',
                    help='random seed')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    device = "cuda:0"
else:
    device = "cpu"

# set up devices, seeds
dtype = torch.float32
torch.manual_seed(args.seed)
if args.cuda:
        torch.cuda.manual_seed(args.seed)
print("Running on device {}".format(device))

# initalize potential, find path end points
from realnvp.two_channel import TwoChannel
target = TwoChannel(device=device)
target.beta = args.beta
target.n_steps = args.bridge_len
target.dt = args.dt

# add a nonequilibrium drift
def curl_drift(xt):
    drift = torch.zeros(xt.shape, device=device)
    drift[:,:,:,0] = -2.5*xt[:,:,:,1]
    drift[:,:,:,1] = 2.5*xt[:,:,:,0]
    return drift

if args.noneq:
    target.drift = curl_drift


n_bridges = args.n_bridges



# set up the Real NVP model
args_target = {
    'path_length': target.n_steps-2,
    'dim': 2,
    'beta': args.beta,
    'dt': args.dt,
    'n_steps': args.bridge_len
}

f = open("target_args.pkl","wb")
pickle.dump(args_target,f)
f.close()

# set up the Brownian bridge prior
bridge_kwargs = target.get_bridge_args()
args_bridge = dict(type="bridge") 
args_bridge["bridge_kwargs"] = bridge_kwargs
print(bridge_kwargs)

args_rnvp = {
    'dim': args_target['path_length'] * args_target['dim'],
    'n_realnvp_block': 10,
    'block_depth': 3,
    'hidden_dim': 100,
    'args_prior': args_bridge,
    'init_weight_scale': 1e-6,
}

f = open("rnvp_args.pkl","wb")
pickle.dump(args_rnvp,f)
f.close()


# set up initial data
from realnvp.two_channel import get_bridge, bridge_energy


from realnvp.sampling import run_action_langevin
dt_path = args.dtp # * target.beta * target.dt
def sample_func(bs, x_init, dtau=dt_path, beta=target.beta):
    xts = run_action_langevin(target, x_init, bs, dtau)
    return xts, xts[-1].detach().requires_grad_()



n_steps = args_bridge["bridge_kwargs"]["n_steps"]
bridges = torch.zeros(n_bridges,n_steps-2,1,2, device=device)
bridges[:,:,0,0] = torch.linspace(-1,1,n_steps)[1:-1]
bridges[:n_bridges//2,:,0,1] = 1.5*torch.cos(0.5*np.pi*torch.linspace(-1,1,n_steps)[1:-1])
bridges = bridges.detach().requires_grad_()

n_burn_in = 2000
_, x_init = sample_func(n_burn_in, bridges)
print(x_init.shape)

# plot the initial data
def plot_paths(x_data, target, step, color="k", alpha=0.1):
    fig, ax, _ = target.plot()
    x_i = x_data.clone().view(n_bridges,target.n_steps-2,2)
    for xt in x_i:
        xs, ys = xt.detach().cpu()[:,0],xt.detach().cpu()[:,1]
        ax.plot(xs,ys, color=color, alpha=alpha)
    fig.savefig("langevin_confs_{:06d}.pdf".format(step))
    plt.close(fig)
    
plot_paths(x_init, target, 0)


def plot_samples(model, target, n_samples, step, alpha=0.1):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

    target.plot(figax=(fig,axs[0]), colorbar=False)

    n_steps = target.n_steps
    bridges = torch.zeros(n_samples,n_steps-2,1,2,device=device)
    for i in range(n_samples):
        bridges[i,:] = get_bridge(**args_bridge["bridge_kwargs"])

    a_min = torch.tensor([bridge_kwargs["x0"],bridge_kwargs["y0"]])
    b_min = torch.tensor([bridge_kwargs["x1"],bridge_kwargs["y1"]])

    gen_samples = model.sample(n_samples)

    for bridge in bridges.clone():
        b = bridge.view(-1,2)
        xs, ys = b.detach().cpu()[:,0],b.detach().cpu()[:,1]
        axs[0].plot(xs,ys,color="k", alpha=0.2)

    target.plot(figax=(fig,axs[1]), colorbar=False)
    for bridge in gen_samples.clone():
        b = bridge.view(-1,2)
        xs, ys = b.detach().cpu()[:,0],b.detach().cpu()[:,1]
        axs[1].plot(xs,ys,color="g", alpha=0.2)
    fig.suptitle("Training step {:d}".format(step))
    fig.tight_layout()
    fig.savefig("sampled_confs_{:06d}.pdf".format(step))
    plt.close(fig)



def nll(x, model, target):
    z, log_det_jac = model.backward(x)
    a_min = torch.tensor([bridge_kwargs["x0"],bridge_kwargs["y0"]], device=device)
    b_min = torch.tensor([bridge_kwargs["x1"],bridge_kwargs["y1"]], device=device)
    betaE = target.beta * bridge_energy(z, dt=target.dt, a_min=a_min, b_min=b_min, device=device)
    return betaE - log_det_jac


def run_action_mh_langevin(model, target, x_init, n_steps, dt_path, resample=False, bc=None):
    '''
    Path action langevin for diffusions
    model (nn.Module): model with the function `model.action(xt)` implemented
    xt (tensor): initial trajectory
    dt (float): integration timestep for the input trajectory
    drift (tensor): non-conservative forcing term with dimensions of x
    bc (float): None or PBC (hypercube) specified as bc=L for [0,L]^d
    '''
    #xts = []
    accs = []
    xt = x_init.clone().detach_().requires_grad_()
        #with torch.no_grad():
    for t in range(n_steps):
        x = model.sample(xt.shape[0])
        xt = xt.detach().clone().reshape(-1, model.dim).requires_grad_()

        ratio = -target.beta * target.U(x) + nll(x, model, target)
        ratio += target.beta * target.U(xt) - nll(xt, model, target)
        ratio = torch.exp(ratio)
        u = torch.rand_like(ratio)
        acc = u < torch.min(ratio, torch.ones_like(ratio))
        if resample:
            x[~acc] = xt[~acc]
            xt.data = x.detach().clone()
        # xs.append(x.clone())
        accs.append(acc)

        gradPath = torch.autograd.grad(target.U(xt).sum(), xt)[0]
        noise = np.sqrt(2 * dt_path / (target.beta)) * \
            torch.randn_like(gradPath)
        xt = xt - gradPath * dt_path + noise
        if bc is not None:
            xt = xt +  bc * (xt < 0.) - bc * (xt > bc)
        #xts.append(xt.clone())
    return xt, torch.stack(accs)


_, x_samples = sample_func(n_bridges, x_init)
x_samples = x_samples.reshape(n_bridges, n_steps-2, 1, 2)


generator = RealNVP_MLP(args_rnvp['dim'], args_rnvp['n_realnvp_block'],
                    args_rnvp['block_depth'], hidden_dim=args_rnvp['hidden_dim'], residual=True, rescale_t=True, 
                    init_weight_scale=args_rnvp['init_weight_scale'],
                    prior_arg=args_rnvp['args_prior'], device=device)


def run_optim_mh(model, target, xi, n_optim_steps=100, lr=1e-3, n_samples=10, checkpoint=1000, resample_threshold=0.1, lossf=None, accsf=None, jump_tol=1e2, js=False, t_start=0.):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    xi = xi.reshape(n_bridges,-1)
    xt = xi.clone()
    last_loss = 0
    lt = 0.
    for step in range(n_optim_steps):
        optimizer.zero_grad()
        x_, acc_ = run_action_mh_langevin(model, target, xt, n_samples, dt_path)
        x = x_.detach().reshape(n_bridges,model.dim).clone().requires_grad_()

        loss = nll(x, model, target).mean()
        loss.backward()
        optimizer.step()

        avg_acc = (1.*acc_).mean().item()
        print(loss.item(), (1.*acc_).mean().item())
        if lossf is not None:
            lossf.write("{:f}\n".format(loss.item()))
            lossf.flush()
        if accsf is not None:
            accsf.write("{:f}\n".format(avg_acc))
            accsf.flush()
        last_loss = loss.clone().detach()
        xt = x.detach().clone().requires_grad_()
        lt += lr
        with torch.no_grad():
            if step%checkpoint==0:
                torch.save(model.state_dict(), 'model_chkpt_{:06d}_nsteps_{:04d}.pkl'.format(step,n_steps))
                plot_samples(generator, target, 100, step)
                plot_paths(x, target, step)


xi_samp = x_samples.detach().clone()
xi_samp.requires_grad_()

loss_file = open("loss.dat", "w")
accs_file = open("accs.dat", "w")

run_optim_mh(generator, target, xi_samp, lr=args.lr, n_samples=args.n_langevin_steps, n_optim_steps=100000, lossf=loss_file, accsf=accs_file, js=args.js)
loss_file.close()
accs_file.close()

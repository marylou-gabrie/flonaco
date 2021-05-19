import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch

from flonaco.gaussian_utils import MoG, plot_2d_level, plot_2d_level_reversed
from flonaco.real_nvp_mlp import RealNVP_MLP
from flonaco.training import train
from flonaco.utils_io import get_file_name

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dtype = torch.float32

data_home = 'temp/'

date = time.strftime('%d-%m-%Y')
random_id = str(np.random.randint(100))
print('random id!', random_id)


parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-dim', '--dim', type=int, default=1)
parser.add_argument('-k', '--k-modes', type=int, default=2)  # k must divide batchsize
parser.add_argument('-cov', '--covariances', type=float, nargs='+', default=1.)
parser.add_argument('-w', '--weights', type=float, nargs='+', default=1.)
parser.add_argument('-sprd', '--spread-from-o', type=float, default=3)
parser.add_argument('-shft', '--shift-from-o', type=float, default=3)

parser.add_argument('-d', '--depth-blocks', type=int, default=2)
parser.add_argument('-hd', '--hidden-dim', type=int, default=10)

parser.add_argument('-samp', '--sampling-method', type=str, default='mhlangevin')
parser.add_argument('-nw', '--n-walkers', type=int, default=100)
parser.add_argument('-dt', '--dt-langevin', type=float, default=1e-4)
parser.add_argument('-rposi', '--ratio-pos-init', type=float, default=[1])

parser.add_argument('-lt', '--loss-type', type=str, default='fwd')  # or 'js', 'bwd'
parser.add_argument('-lr', '--learning-rate', type=float,  default=1e-2)
parser.add_argument('-niter', '--n-iter', type=int,  default=int(1e1))
parser.add_argument('-bs', '--batch-size', type=int, default=int(1e3))
parser.add_argument('-id', '--slurm-id', type=str, default=str(random_id))

parser.add_argument('-sp', '--save-splits', type=int, default=10)

args = parser.parse_args()

dim = args.dim
k = args.k_modes
thetas = np.linspace(0, 2 * np.pi, k + 1)[:-1]
means = [args.spread_from_o * torch.tensor([np.cos(t),np.sin(t)], dtype=dtype, device=device) + args.shift_from_o for t in thetas]

if len(args.covariances) == 1:
    covars = [args.covariances[0] * torch.eye(dim, device=device, dtype=dtype)] * k
elif type(args.covariances) == list:
    covars = [c * torch.eye(dim, device=device, dtype=dtype) for c in args.covariances] 
else:
    raise NotImplemented

if len(args.weights) == 1:
    weights = args.weights * k
elif type(args.weights) == list:
    weights = args.weights
else:
    raise NotImplemented

mog = MoG(means, covars, weights=weights, dtype=dtype, device=device)
Us_g = plot_2d_level(mog, x_min=-10, x_max=10, n_points=100)

args_target = {
    'type': 'mog',
    'dim': args.dim,
    'means': mog.means,
    'covars': mog.covars,
    'weights': mog.weights
}

args_rnvp = {
    'dim': args_target['dim'],
    'n_realnvp_block': args.depth_blocks,
    'block_depth': 1,
    'hidden_dim': args.hidden_dim,
    'args_prior': {'type':'standn'},
    'init_weight_scale': 1e-6,
}

args_training = {
    'args_losses': [
        {'type': args.loss_type, 
        'samp': args.sampling_method, 
        'dt': args.dt_langevin, 
        'beta': 1.0,
        'n_tot': args.n_walkers,
        'ratio_pos_init': args.ratio_pos_init,
        'n_steps_burnin': 1e2
        }
        ],
    'args_stops': [
        {'acc': None}
        ], 
    'n_iter': args.n_iter,
    'lr': args.learning_rate, 
    'bs': args.batch_size 
}

model = RealNVP_MLP(args_rnvp['dim'], args_rnvp['n_realnvp_block'],
                    args_rnvp['block_depth'], 
                    hidden_dim=args_rnvp['hidden_dim'],
                    init_weight_scale=args_rnvp['init_weight_scale'],
                    prior_arg=args_rnvp['args_prior'],
                    device=device)

model_init = copy.deepcopy(model)

x_init_samp = None
for args_loss, args_stop in zip(args_training['args_losses'], args_training['args_stops']):
    args_loss['x_init_samp'] = x_init_samp
    _ = train(model, mog, n_iter=args_training['n_iter'], 
              lr=args_training['lr'], bs=args_training['bs'],
              args_loss=args_loss, args_stop=args_stop,
              estimate_tau=True,
              return_all_xs=(args.n_iter * args.batch_size < 1e7),
              save_splits=args.save_splits
              )
    to_return = _  
    xs = to_return['xs']  
    # xs are all the langein samples, list of the len n_iter or save_splits
    # each element is an array of shape (n_steps, n_tot, dim)
    x_init_samp = xs[-1].reshape(-1, model.dim)


results = {
    'args_target': args_target,
    'args_model': args_rnvp, 
    'args_training': args_training,
    'target': mog,
    'model_init': model_init, 
    'model': model,
    'final_xs': xs[-1], # save last samples to reuse as starting points
    'xs' : to_return['xs'],
    'models' : to_return['models'], 
    'losses': to_return['losses'],
    'acc_rates': to_return['acc_rates'],
    'taus': to_return['taus']
}

filename = get_file_name(args_target, args_rnvp, args_training,
                            date=date, random_id=args.slurm_id,
                            data_home=data_home)
torch.save(results, filename)
print('saved in:', filename)
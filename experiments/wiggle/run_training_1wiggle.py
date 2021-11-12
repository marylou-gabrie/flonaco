import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn.functional as F

from flonaco.croissant_utils import Croissants, plot_2d_level
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


parser.add_argument('-d', '--depth-blocks', type=int, default=2)
parser.add_argument('-hdm', '--hidden-dim', type=int, default=100)
parser.add_argument('-hdp', '--hidden-depth', type=int, default=3)
parser.add_argument('-hda', '--hidden-activation', type=str, default='relu')
parser.add_argument('-mdt', '--models-type', type=str, default='mlp')
parser.add_argument('--hidden-bias', default=False, action='store_true') 
parser.add_argument('-mt', '--mask-type', type=str, default='inter')
parser.add_argument('--last-scale', default=False, action='store_true')
parser.add_argument('-pt', '--prior-type', type=str, default='white')
parser.add_argument('-pic', '--prior-init-cov', type=float, default=1.)

parser.add_argument('-samp', '--sampling-method', type=str, default='mhlangevin')
parser.add_argument('-nw', '--n-walkers', type=int, default=100)
parser.add_argument('-dt', '--dt-langevin', type=float, default=1e-4)
parser.add_argument('-rposi', '--ratio-pos-init', type=float, default=[1])

parser.add_argument('-lt', '--loss-type', type=str, default='fwd')
parser.add_argument('-lr', '--learning-rate', type=float,  default=1e-2)
parser.add_argument('-niter', '--n-iter', type=int,  default=int(1e1))
parser.add_argument('-bs', '--batch-size', type=int, default=int(1e3))
parser.add_argument('-id', '--slurm-id', type=str, default=str(random_id))
parser.add_argument('--schedule', default=False, action='store_true')

parser.add_argument('-sp', '--save-splits', type=int, default=10)

args = parser.parse_args()

dim = 2
k = 1
means = [torch.tensor([6, 0], dtype=dtype, device=device),]
cv = 1 * torch.eye(dim, dtype=dtype, device=device)
covars = [cv.clone() for c in range(1)]
weights = [1]

ring_mean = 5
ring_var = 8


crst = Croissants(means, covars, ring_mean, 
                  ring_var, weights=weights, 
                  wiggle=True,
                  dtype=dtype, device=device)

Us_g = plot_2d_level(crst, x_min=-10, x_max=10, n_points=100)

args_target = {
    'type': 'croissant',
    'dim': dim, 
    'means': crst.means, 
    'covars': crst.covars,  
    'weights': crst.weights, 
    'ring_mean': crst.ring_mean,  
    'ring_var': crst.ring_var, 
    'wiggle': crst.wiggle
}

args_rnvp = {
    'dim': dim,
    'n_realnvp_block': args.depth_blocks,
    'block_depth': 1,
    'hidden_dim': args.hidden_dim,
    'hidden_depth': args.hidden_depth,
    'hidden_bias': args.hidden_bias,
    'hidden_activation': args.hidden_activation,
    'args_prior': {'type': args.prior_type,
                   },
    'init_weight_scale': 1e-6,
    'models_type': args.models_type,
    'mask_type': args.mask_type,
    # 'dim_phys': args_target['dim_phys'],
    'last_scale': args.last_scale
}

args_training = {
    'args_losses': [
        {'type': args.loss_type, 
        'samp': args.sampling_method, 
        'dt': args.dt_langevin, 
        'beta': 1.0,
        'n_tot': args.n_walkers,
        'ratio_pos_init': args.ratio_pos_init,
        'n_steps_burnin': 1e0
        }
        ],
    'args_stops': [
        {'acc': None}
        ], 
    'n_iter': args.n_iter,
    'lr': args.learning_rate, 
    'bs': args.batch_size 
}

cov = args.prior_init_cov * torch.eye(dim, dtype=dtype, device=device)
mean = torch.zeros(dim, dtype=dtype, device=device)

args_rnvp['args_prior']['cov'] = cov
args_rnvp['args_prior']['mean'] = mean

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
    _ = train(model, crst, n_iter=args_training['n_iter'], 
              lr=args_training['lr'], bs=args_training['bs'],
              args_loss=args_loss, args_stop=args_stop,
              estimate_tau=True,
              return_all_xs=True,
              save_splits=args.save_splits,
              use_scheduler=args.schedule,
              step_schedule=int(args.n_iter/4)
              )
    to_return = _  
    xs = to_return['xs']  
    # xs are all the langein samples, list of the len n_iter
    # each element is an array of shape (n_steps, n_tot, dim)
    x_init_samp = xs[-1].reshape(-1, model.dim)
    
    
results = {
    'args_target': args_target,
    'args_model': args_rnvp, 
    'args_training': args_training,
    'target': crst,
    'model_init': model_init, 
    'model': model,
    'final_xs': xs[-1], # save last langevin samples to reuse as starting points
    'xs': to_return['xs'],
    'models': to_return['models'], 
    'losses': to_return['losses'],
    'acc_rates': to_return['acc_rates'],
    'acc_rates_mala': to_return['acc_rates_mala'],
    'taus': to_return['taus'],
    'grad_norms': to_return['grad_norms'],
}

filename = get_file_name(args_target, args_training, args_model=args_rnvp,
                            date=date, random_id=args.slurm_id,
                            data_home=data_home)
torch.save(results, filename)
print('saved in:', filename)

plt.show(block=False)
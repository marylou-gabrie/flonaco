import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import time
import torch

from flonaco.phifour_utils import PhiFour
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
parser.add_argument('-N', '--N', type=int, default=10)
parser.add_argument('-dp', '--dim-phys', type=int, default=1)
parser.add_argument('-pt', '--prior-type', type=str, default='coupled')
parser.add_argument('-pa', '--prior-a', type=float, default=1)

parser.add_argument('-a', '--a-coupling', type=float, default=0.1)
parser.add_argument('-b', '--b-field', type=float, default=0)
parser.add_argument('-beta', '--beta', type=float, default=20)
parser.add_argument('-tv', '--tilt-value', type=float, default=None)
parser.add_argument('-tlbd', '--tilt-lambda', type=float, default=None)

parser.add_argument('-d', '--depth-blocks', type=int, default=2)
parser.add_argument('-hd', '--hidden-dim', type=int, default=100)
parser.add_argument('-mt', '--mask-type', type=str, default='half')

parser.add_argument('-samp', '--sampling-method', type=str, 
                    default='mhlangevin')
parser.add_argument('-nw', '--n-walkers', type=int, default=100)
parser.add_argument('-dt', '--dt-langevin', type=float, default=1e-4)
parser.add_argument('-rposi', '--ratio-pos-init', type=float, default=0.5) # -1 for 'randn'

parser.add_argument('-lt', '--loss-type', type=str, default='fwd') # or 'js'
parser.add_argument('-lr', '--learning-rate', type=float,  default=1e-2)
parser.add_argument('-niter', '--n-iter', type=int,  default=int(1e1))
parser.add_argument('-bs', '--batch-size', type=int, default=int(1e3))
parser.add_argument('-id', '--slurm-id', type=str, default=str(random_id))

args = parser.parse_args()

if args.tilt_value is not None:
    tilt = {'val': args.tilt_value, 'lambda': args.tilt_lambda}
else:
    tilt = None

args_target = {
    'type': 'phi4',
    'N': args.N,  # size of the grid along one dimension
    'dim_phys': args.dim_phys,  # only 1 implemented
    'a': args.a_coupling,  # coupling
    'b': args.b_field,  # bias
    'beta': args.beta,  # inverse temp
    'tilt': tilt # tilt mean of the distribution
}

args_rnvp = {
    'dim': args_target['N'] * args_target['dim_phys'],
    'n_realnvp_block': args.depth_blocks,
    'block_depth': 1,
    'hidden_dim': args.hidden_dim,
    'args_prior': {'type': args.prior_type,
                    'alpha': args_target['a'],
                   'beta': args_target['beta']},
    'init_weight_scale': 1e-6,
    'mask_type': args.mask_type
}

args_training = {
    'args_losses': [
        {'type': args.loss_type, 
        'samp': args.sampling_method, 
        'dt': args.dt_langevin, 
        'beta': 1.0,
        'n_tot': args.n_walkers,
        'ratio_pos_init': 'rand' if args.ratio_pos_init == -1 else args.ratio_pos_init,
        'n_steps_burnin': 1e3
        }
        ],
    'args_stops': [
        {'acc': 1.0}
        ], 
    'n_iter': args.n_iter,
    'lr': args.learning_rate, 
    'bs': args.batch_size 
}

phi4 = PhiFour(args_target['a'], args_target['b'], 
               args_target['N'], 
               dim_phys=args_target['dim_phys'],
               beta=args_target['beta'],
               tilt=tilt)

model = RealNVP_MLP(args_rnvp['dim'], args_rnvp['n_realnvp_block'],
                    args_rnvp['block_depth'], 
                    hidden_dim=args_rnvp['hidden_dim'],
                    init_weight_scale=args_rnvp['init_weight_scale'],
                    prior_arg=args_rnvp['args_prior'],
                    mask_type=args_rnvp['mask_type'],
                    device=device)

model_init = copy.deepcopy(model)

x_init_samp = None
for args_loss, args_stop in zip(args_training['args_losses'], args_training['args_stops']):
    args_loss['x_init_samp'] = x_init_samp
    _ = train(model, phi4, n_iter=args_training['n_iter'], 
              lr=args_training['lr'], bs=args_training['bs'],
              args_loss=args_loss, args_stop=args_stop,
              estimate_tau=True,
              return_all_xs=(args.n_iter * args.batch_size < 1e7)
              )
    to_return = _  
    xs = to_return['xs']  
    # xs are all the langein samples, list of the len n_iter
    # each element is an array of shape (n_steps, n_tot, dim)
    x_init_samp = xs[-1].reshape(args_training['bs'], model.dim)
    
    
results = {
    'args_target': args_target,
    'args_model': args_rnvp, 
    'args_training': args_training,
    'target': phi4,
    'model_init': model_init, 
    'model': model,
    'final_xs': xs[-1], # save last langevin samples to reuse as starting points
    'xs' : to_return['xs'],
    'models' : to_return['models'], 
    'losses': to_return['losses'],
    'acc_rates': to_return['acc_rates'],
    'taus': to_return['taus']
}

filename = get_file_name(args_target, args_training, args_model=args_rnvp, 
                            date=date, random_id=args.slurm_id,
                            data_home=data_home)
torch.save(results, filename)
print('saved in:', filename)
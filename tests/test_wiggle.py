import copy
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.shape_base import block
import torch
import time
from flonaco.real_nvp_mlp import RealNVP_MLP
from flonaco.training import train 
from flonaco.croissant_utils import (
    Croissants, plot_2d_level
)

torch.manual_seed(121)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

date = time.strftime('%d-%m-%Y')
random_id = str(np.random.randint(100))
print('random id!', random_id)


dim = 2
k = 2
means = [torch.tensor([6, 0], dtype=dtype),]
cv = 1 * torch.eye(dim, dtype=dtype)
covars = [cv.clone() for c in range(1)]
weights = [1]

ring_mean = 5
ring_var = 8

crst = Croissants(means, covars, ring_mean,
                  ring_var, weights=weights,
                  dtype=dtype, device=device,
                  wiggle=True)

plt.figure(figsize=(4, 4))
ax1 = plt.subplot(111)
Us_g = plot_2d_level(crst, x_min=-10, x_max=10, n_points=100, ax=ax1)
plt.show(block=False)


args_target = {
    'type': 'croissants',
    'dim': crst.dim,
    'means': crst.means,
    'covars': crst.covars,
    'ring_mean': crst.ring_mean,
    'ring_var': crst.ring_var,
    'weights': crst.weights
}

cov = 0.2 * torch.eye(dim, dtype=dtype, device=device)
mean = torch.zeros(dim, dtype=dtype, device=device)

args_rnvp = {
    'dim': args_target['dim'],
    'n_realnvp_block': 2,
    'block_depth': 1,
    'args_prior': {'type': 'white',  'cov': cov, 'mean': mean},
    'init_weight_scale': 1e-6,
}

args_training = {
    'args_losses': [
    {
    'type': 'fwd',
    'samp': 'mh', 'dt': 1e-4, 
    'beta': 1.0, 'n_steps_burnin': int(1),
    'n_tot': 100, 
    }
    ],
    'args_stops': [
        {'acc': 1.0}
        ], 
    'n_iter': int(5),
    'lr': 1e-2, 
    'bs': int(5e2) 
}

model = RealNVP_MLP(args_rnvp['dim'],
                    args_rnvp['n_realnvp_block'],
                    args_rnvp['block_depth'],
                    init_weight_scale=args_rnvp['init_weight_scale'],
                    prior_arg=args_rnvp['args_prior'],
                    device=device)

model_init = copy.deepcopy(model)

x_init_samp = None
for args_loss, args_stop in zip(args_training['args_losses'],
                                args_training['args_stops']):

    args_loss['x_init_samp'] = x_init_samp
    _ = train(model, crst, n_iter=args_training['n_iter'], 
              lr=args_training['lr'], bs=args_training['bs'],
              args_loss=args_loss, args_stop=args_stop
              )
    to_return = _  
    xs = to_return['xs']
    x_init_samp = xs[-1].reshape(args_training['bs'], model.dim)

results = {
    'args_target': args_target,
    'args_model': args_rnvp, 
    'args_training': args_training,
    'target': crst,
    'model_init': model_init, 
    'model': model,
    'final_xs': xs[-1],  # save last samples to reuse as starting points
    'xs': to_return['xs'],
    'models': to_return['models'], 
    'losses': to_return['losses'],
    'acc_rates': to_return['acc_rates']
}

plt.figure(figsize=(12, 4))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)
Us_rnvp = plot_2d_level(model_init, x_min=-10, x_max=10, n_points=100, ax=ax1)
Us_rnvp = plot_2d_level(model, x_min=-10, x_max=10, n_points=100, ax=ax2)
Us_g = plot_2d_level(crst, x_min=-10, x_max=10, n_points=100, ax=ax3)

plt.show(block=False)
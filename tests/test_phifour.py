import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from flonaco.phifour_utils import PhiFour
from flonaco.real_nvp_mlp import RealNVP_MLP
from flonaco.training import train
from flonaco.utils_plots import (
    plot_map_point_cloud, 
    plot_map_point_cloud_Fourier,
    plot_Fourier_spectrum
)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

N = 10 
dim_phys = 1
a = 0.1
b = 0.0
beta = 10

### DEFINE MODEL

phi4 = PhiFour(a, b, N, dim_phys=dim_phys, beta=beta, 
# tilt={'val':0.7,'lambda':100},
)

### RUN TRAINING
dim = N * dim_phys
n_realnvp_block = 2
block_depth = 1

model = RealNVP_MLP(dim, n_realnvp_block, block_depth,
                    init_weight_scale=1e-3,
                    prior_arg={'type': 'coupled', 'alpha': a, 'beta': beta},
                    mask_type='half',
                    )

args_loss = {'type': 'fwd',
    'samp': 'mhlangevin', 'dt': 1e-4, 'beta': 1.0, 'n_steps_burnin': 1e2,
    'n_tot': 100,  'ratio_pos_init': 0.5}

args_stop = {'acc': 1.0}
bs = int(1e3)

x_init_samp = None
args_loss['x_init_samp'] = x_init_samp

_ = train(model, phi4, n_iter=int(1e1), lr=1e-2, bs=bs,
    args_loss=args_loss, args_stop=args_stop
        )

model = _['models'][-1]
xs = _['xs']
x_init_samp = xs[-1].reshape(bs, model.dim)

plt.show(block=False)
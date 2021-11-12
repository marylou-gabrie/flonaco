import copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import torch
from torch.nn.utils import clip_grad_norm_
import flonaco.gaussian_utils
import flonaco.phifour_utils
from flonaco.phifour_utils import PhiFour
from flonaco.gaussian_utils import MoG
from flonaco. croissant_utils import Croissants
from flonaco.sampling import (
    run_langevin,
    run_MALA,
    run_metropolis,
    run_metrolangevin,
    run_metromalangevin,
    run_action_langevin,
    estimate_deltaF,
    compute_ESS
)


def train(model, target, n_iter=10, lr=1e-1, bs=100,
          use_scheduler=False,
          step_schedule=10000,
          args_loss={'type': 'fwd', 'samp': 'direct'},
          args_stop={'acc': None},
          estimate_tau=False,
          return_all_xs=True,
          jump_tol=1e2,
          save_splits=10,
          grad_clip=1e4):
    """"
    Main training/sampling function.

    Args:
        model (Realnvp_MLP)
        target (MoG, PhiFour)
        n_iter (int)
        lr (float): learning rate
        bs (int): batchsize
        use_scheduler (bool): if learning rate schedule should be used
        step_schedule (int): iteration frequency of schedule   
        args_loss (dict): 
                    'type' - loss type 'fwd', 'bwd', 'js'
                    'samp' - sampling method 'langevin', 'direct', 'mhlangevin' 
                    + kwargs for sampling method
                    Note that not all combinations are possible
                    depending on target etc.
        args_stop (dict): {'acc': x} with x in [0,1] Metropolis acceptance
                    threshold to stop train
        estimate_tau: estimates autocorrelation time
        return_all_xs: will return samples produced on the fly
        jump_tol: will terminate learning if loss jumps above in one iteration
        save_splits: number of snapshots saved during training
    """

    # setting up the loss
    if args_loss['type'] in ['fwd', 'js']:
        def loss_func(x): return (model.nll(x) - target.U(x)).mean()
    elif args_loss['type'] == 'bwd':
        def loss_func(x): return (- model.nll(x) + target.U(x)).mean()
    else:
        raise NotImplementedError("This loss type is not available.")


    # setting the sampling
    if args_loss['samp'] == 'direct':
        if args_loss['type'] in ['fwd', 'js']:
            assert isinstance(target, MoG)
            def sample_func(bs): return target.sample(bs)
            kwargs = {}
        elif args_loss['type'] in ['bwd']:
            def sample_func(bs): return model.sample(bs)
            kwargs = {}
    elif args_loss['samp'] == 'mh':
        if args_loss['x_init_samp'] is None:
            x_init = model.sample(args_loss['n_tot'])
        else:
            x_init = args_loss['x_init_samp'][:args_loss['n_tot']]

        def sample_func(bs, x_init=x_init, acc_rate=None):
            n_steps = int(bs / x_init.shape[0])
            x, acc = run_metropolis(
                model, target, x_init, n_steps)
            kwargs['x_init'] = x[-1, ...]
            
            kwargs['acc_rate'] = (acc.cpu().numpy() * 1).mean()
            return x

        kwargs = {'x_init': x_init}


    
    elif 'langevin' in args_loss['samp']:
        ## setting initialization for chain methods
        skip_burnin = False
        assert args_loss['n_tot'] <= bs
        
        if args_loss['x_init_samp'] is not None:
            x_init = args_loss['x_init_samp'][-args_loss['n_tot']:]
            skip_burnin = True
        elif args_loss['ratio_pos_init'] == 'rand':
            x_init = torch.randn(args_loss['n_tot'], model.dim,
                                 device=model.device)
            print('Random init!')
        elif isinstance(target, MoG) or isinstance(target, Croissants):
            x_init = torch.stack(target.means)
            x_init = x_init.repeat_interleave(
                int(args_loss['n_tot'] / len(target.means)), dim=0)
        elif isinstance(target, PhiFour):
            if 'n_tot' in args_loss.keys():
                x_init = torch.ones(args_loss['n_tot'], model.dim, 
                                    device=model.device)
                n_pos = int(args_loss['ratio_pos_init'] * args_loss['n_tot'])
                if target.tilt is None:
                    x_init[n_pos:, :] = -1
                else:
                    n_tilt = int(target.tilt['val'] * model.dim)
                    x_init[n_pos:, n_tilt:] = -1
                    x_init[:n_pos, :(model.dim - n_tilt)] = -1
            else:
                raise RuntimeError('Could not understand Langevin init')

        else:
            raise NotImplementedError("That target class is not supported")

        x_init = x_init.detach().requires_grad_()

        ## setting samplimg functions
        if args_loss['samp'] == 'langevin':

            def sample_func(bs, x_init=x_init, dt=100, beta=1):
                n_steps = int(bs / x_init.shape[0])
                x = run_langevin(target, x_init, n_steps, dt * model.dim)
                kwargs['x_init'] = x[-1, ...].detach().requires_grad_()
                return x

        elif args_loss['samp'] == 'mhlangevin':

            def sample_func(bs, x_init=x_init, dt=100, beta=1, acc_rate=None):
                n_steps = int(bs / x_init.shape[0])
                x, acc = run_metrolangevin(
                    model, target, x_init, n_steps, dt * model.dim)
                kwargs['x_init'] = x[-1, ...].detach().requires_grad_()
                kwargs['acc_rate'] = (acc.cpu().numpy() * 1).mean()
                return x
        elif args_loss['samp'] == 'mhmalangevin':

            def sample_func(bs, x_init=x_init, dt=100, beta=1, alpha=0, acc_rate=None):
                n_steps = int(bs / x_init.shape[0])
                # x, acc = run_metrolangevin_dirpas(
                x, acc = run_metromalangevin(
                    model, target, x_init, n_steps, dt * model.dim)
                kwargs['x_init'] = x[-1, ...].detach().requires_grad_()
                kwargs['acc_rate'] = (acc.cpu().numpy() * 1).mean()
                return x
    
        elif args_loss['samp'] == 'malangevin':

            def sample_func(bs, x_init=x_init, dt=100, beta=1, acc_rate=None):
                n_steps = int(bs / x_init.shape[0])
                # x, acc = run_metrolangevin_dirpas(
                x, acc = run_MALA(
                    target, x_init, n_steps, dt=dt * model.dim)
                kwargs['x_init'] = x[-1, ...].detach().requires_grad_()
                kwargs['acc_rate'] = (acc.cpu().numpy() * 1).mean()
                return x

        elif args_loss['samp'] == 'langevin_action':

            def sample_func(bs, x_init, dt=5e-9, beta=target.beta):
                drift = torch.zeros(1,2)
                xts = run_action_langevin(target, x_init, bs, 5e-9, target.dt,  bc=1.0)
                kwargs['x_init'] = xts[-1, ...].detach().requires_grad_()
                return xts

        kwargs = {'x_init': x_init,
                  'dt': args_loss['dt'],
                  'beta': args_loss['beta']}

        if not skip_burnin:
            bs_burnin = int(args_loss['n_steps_burnin'] * x_init.shape[0])
            start = time.time()
            n_steps = int(bs_burnin / x_init.shape[0])
            x = run_langevin(target, x_init, n_steps, args_loss['dt'] * model.dim)
            kwargs['x_init'] = x[-1, ...].detach().requires_grad_()
            print('Langevin burnin done! time: {:f}s'.format(time.time() - start))
            print('Burn-in done!')

    assert sample_func is not None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=step_schedule, 
                                                    gamma=0.5)

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(2, 5)
    axs = [fig.add_subplot(gs[a]) for a in range(10)]
    a = 0  # counter index for axs

    # logs
    xs = []
    losses = []
    models = [copy.deepcopy(model)]
    taus = []
    acc_rates = []
    acc_rates_mala = []
    grad_norms = []

    for t in range(n_iter):
        optimizer.zero_grad()

        x_ = sample_func(bs, **kwargs)

        x = x_.reshape(-1, model.dim).detach().requires_grad_()
        loss = loss_func(x)
        if args_loss['samp'] == 'langevin_action':
            loss = loss * target.dt * target.beta

        if t > 0 and loss - losses[-1] > jump_tol:
            print('t = {:d}, KL wants to jump, resampling'.format(t))
            for trial in range(5):
                x_ = sample_func(bs, **kwargs)
                x = x_.reshape(-1, model.dim).detach().requires_grad_()
                loss = loss_func(x)

                if loss - losses[-1] <= jump_tol:
                    break

            if loss - losses[-1] > jump_tol:
                print('KL wants to jump, terminating learning')
                break

        if return_all_xs or t % (n_iter / 10) == 0:
            xs.append(x_)

        if args_loss['type'] == 'js':
            xb = model.sample(bs)
            loss += (- model.nll(xb) + target.U(xb)).mean()

        if torch.isinf(loss).any():
            print('Stopped because loss became inf!')
            return model, losses, xs

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        losses.append(loss.item())

        if t % (n_iter / 100) == 0:
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5 
            grad_norms.append(total_norm)

        if use_scheduler:
            scheduler.step()

        if estimate_tau:
            tau = x.shape[0] * x.shape[1] / \
                np.mean(compute_ESS(x.detach().cpu()))
            taus.append(tau)

        x_last = x.clone()
        _, acc = run_metropolis(model, target, x_last, 1)
        acc_rate = (acc.cpu().numpy() * 1).mean()
        acc_rates.append(acc_rate.item())
        _, acc = run_MALA(target, x_last, 1,  dt=args_loss['dt'] * model.dim)
        acc_rate = (acc.cpu().numpy() * 1).mean()
        acc_rates_mala.append(acc_rate.item())

        #prints
        if t % (n_iter / save_splits) == 0 or n_iter <= save_splits:
            models.append(copy.deepcopy(model))
            print('t={:0.1e}'.format(t),
                  'Loss: {:3.2f}'.format(loss.item()), end='  \t')

            print('mh acc: {:0.2e}, mala acc: {:0.2e}'.format(acc_rates[-1],
                                                      acc_rates_mala[-1]),
                                                      end='\t')

            print('Gd: {:0.0e}'.format(total_norm), end='\t')

            for param_group in optimizer.param_groups:
                print('lr: {:0.2e}'.format(param_group['lr']), end='\t')

                # compute fraction with mean > 0 or mean < 1
            if isinstance(target, PhiFour):
                x_gen = model.sample(xs[-1].shape[1])
                frac_pos = (x_gen.mean(1) > 0).sum() / float(x_gen.shape[0])
                print('Frac gen pos: {:0.2f}'.format(frac_pos.item()))

            print('')

        if t % (n_iter / 10) == 0:
            if model.dim == 2:
                assert isinstance(target, MoG) or isinstance(target, Croissants)
                x_min = -10 if isinstance(target, MoG) else -0.5
                x_max = 10 if isinstance(target, MoG) else 0.5
                flonaco.gaussian_utils.plot_2d_level(model, ax=axs[a],
                                                    title='t= ' + str(t),
                                                    x_min=x_min, x_max=x_max)

                plt.scatter(x[:, 0].detach().cpu(),
                            x[:, 1].detach().cpu(), s=1., alpha=0.1)

            else:
                if (isinstance(target, PhiFour)) and target.dim_phys == 1:
                    plt.sca(axs[a])
                    plt.title('t= ' + str(t))
                    x_gen = model.sample(xs[-1].shape[1])
                    for i in range(xs[-1].shape[1]):
                        plt.plot(xs[-1][-1,  i, :].detach().cpu(),
                                 c='b', alpha=0.2)
                    for i in range(x_gen.shape[0]):
                        plt.plot(x_gen[i, :].detach().cpu(),
                                 c='k', alpha=0.2)

            plt.tight_layout()
            a += 1  # update counter of axes to plots

    to_return = {
        'model': model,
        'losses': losses,
        'xs': xs,
        'models': models,
        'taus': taus,
        'acc_rates': acc_rates,
        'acc_rates_mala': acc_rates_mala,
        'grad_norms': grad_norms,
    }

    return to_return

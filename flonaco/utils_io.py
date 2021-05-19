data_home = 'temp/'

def get_file_name(args_target, args_model, 
                  args_training, 
                  date='', 
                  random_id='',
                  data_home=data_home):
    folder = data_home + 'models/' + args_target['type']

    n_realnvp_block, block_depth = (args_model[key] for key in ['n_realnvp_block', 'block_depth'])

    name = date + '_'+args_training['args_losses'][0]['samp']+'_' + args_target['type'] 

    if args_target['type'] == 'phi4':
        N, a, b, beta, tilt = (args_target[key] for key in ['N', 'a', 'b', 'beta', 'tilt'])
        langevin_ratio_pos_init = args_training['args_losses'][0]['ratio_pos_init']
        prior = args_model['args_prior']['type']

        name += '_N{:d}_a{:0.2f}_b{:0.2e}_beta{:0.2f}'.format(N, a, b, beta)
        if tilt is not None:
            name += '_tv{:0.2f}'.format(tilt['val'])
        name += '_prior{:s}'.format(prior)
        name += '_rposinit{:s}'.format(str(langevin_ratio_pos_init))
    elif args_target['type'] == 'mog':
        dim, k = args_target['dim'], len(args_target['means'])
        name += '_{:d}D_k{:d}'.format(dim, k)
    name += '_{:d}blocks{:d}deep_{:s}.pyT'.format(n_realnvp_block, block_depth, random_id)

    return folder + '/' + name
data_home = 'temp/'

def get_file_name(args_target,  
                  args_training, 
                  args_model=None,
                  date='', 
                  random_id='',
                  plus='',
                  data_home=data_home):
    folder = data_home + 'models/' + args_target['type']
    
    name = date + '_' + args_training['args_losses'][0]['samp'] + '_'
    name += args_target['type'] 
    
    if args_target['type'] == 'phi4':
        N, a, b, beta, tilt = (args_target[key] for key in ['N', 'a', 'b',
                                                            'beta', 'tilt'])
        lang_ratio_pos_init = args_training['args_losses'][0]['ratio_pos_init']
        prior = args_model['args_prior']['type']
        name += '_N{:d}_a{:0.2f}_b{:0.2e}_beta{:0.2f}'.format(N, a, b, beta)
        if tilt is not None:
            name += '_tv{:0.2f}'.format(tilt['val'])
        name += '_prior{:s}'.format(prior)

        name += '_rposinit{:s}'.format(str(lang_ratio_pos_init))

    elif args_target['type'] == 'mog':
        dim, k = args_target['dim'], len(args_target['means'])
        name += '_{:d}D_k{:d}'.format(dim, k)

    elif args_target['type'] == 'croissant':
        dim, k = args_target['dim'], len(args_target['means'])
        c = args_model['args_prior']['cov'][0,0]
        name += '_{:d}D_k{:d}_pic{:0.2e}'.format(dim, k, c)
        if args_target['wiggle']:
            name += '_wiggle'
    
    if args_model is not None:
        r_bck, bck_d = (args_model[key] for key in ['n_realnvp_block',
                                                    'block_depth'])
        name += '_{:d}blocks{:d}deep'.format(r_bck, bck_d)

    name += '_{:s}_{:s}.pyT'.format(plus, random_id)

    return folder + '/' + name

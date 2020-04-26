import os
import argparse
import torch
import utils

from os import listdir, path, mkdir


def collect_args_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', 
                        choices=[
                                 'baseline' 
                                ])
    
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.set_defaults(cuda=True)

    opt = vars(parser.parse_args())
    opt = create_experiment_setting(opt)
    return opt

def create_experiment_setting(opt):
    opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
    opt['dtype'] = torch.float32
    opt['print_freq'] = 50
    opt['total_epochs'] = 20
    opt['output_dim']=101
    opt['save_folder'] = os.path.join('record/'+opt['experiment'])
    utils.make_dir('record')
    utils.make_dir('record/'+opt['experiment'])
    
    optimizer_setting = {
        'optimizer': torch.optim.Adam,
        'lr': 1e-4,
        'weight_decay': 0,
    }
    opt['optimizer_setting'] = optimizer_setting
    opt['dropout'] = 0.5
    
    if opt['experiment']=='baseline':
        
        params_train = {'batch_size': 32,
                 'shuffle': True,
                 'num_workers': 2}
        
        params_val = {'batch_size': 32,
                 'shuffle': True,
                 'num_workers': 2}

            
        data_setting = {
            'path': '/n/fs/visualai-scr/vramaswamy/COS529_project/data/UCF-101',
            'train_params': params_train,
            'test_params': params_val,
        }
        opt['data_setting'] = data_setting
    
        
    return opt



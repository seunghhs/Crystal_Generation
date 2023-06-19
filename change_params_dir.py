

import os
import torch
from copy import deepcopy


tmp = torch.load('./generation/epoch=888-step=88888.ckpt')

## change the dir_name of hyper_parameters

new_hyper_dict = deepcopy(tmp['hyper_parameters'])


new_path = os.getcwd() + '/cdvae/'
new_hyper_dict['decoder']['scale_file'] = new_path + 'cdvae/pl_modules/gemnet/gemnet-dT.json'
new_hyper_dict['data']['root_path'] = new_path + 'data/mp_20'
new_hyper_dict['data']['datamodule']['datasets']['train']['path'] = new_path + 'data/mp_20/train.csv'
new_hyper_dict['data']['datamodule']['datasets']['val'][0]['path'] = new_path +'data/mp_20/val.csv'
new_hyper_dict['data']['datamodule']['datasets']['test'][0]['path'] = new_path + 'data/mp_20/test.csv'
#new_hyper_dict

## change the hydra_dir_name of callbacks
new_callback= deepcopy(tmp['callbacks'])
new_model_name = './generation/epoch=888-step=88888.ckpt'
new_hydra_path = os.getcwd() + '/generation/'
key = list(new_callback.keys())[1]
new_callback[key]['best_model_path'] = new_hydra_path + new_model_name
new_callback[key]['dirpath'] = new_hydra_path

tmp['callbacks'] = new_callback
tmp['hyper_parameters'] = new_hyper_dict

torch.save(tmp, f'{new_model_name}')
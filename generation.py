#### code modefied from CDVAE code.

import os
import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch

import hydra
from hydra import initialize_config_dir
from hydra.experimental import compose
import numpy as np
from cdvae.common.data_utils import chemical_symbols
from torch.nn import functional as F
from torch_scatter import scatter
import json
import numpy as np


def load_model(model_path, load_data=False, testing=True):
    #print(model_path)
    with initialize_config_dir(str(model_path)):
        cfg = compose(config_name='hparams')
        model = hydra.utils.instantiate(
            cfg.model,
            optim=cfg.optim,
            data=cfg.data,
            logging=cfg.logging,
            _recursive_=False,
        )
        ckpts = list(model_path.glob('*.ckpt'))
        if len(ckpts) > 0:
            ckpt_epochs = np.array(
                [int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        model = model.load_from_checkpoint(ckpt)
        model.lattice_scaler = torch.load(model_path / 'lattice_scaler.pt')
        model.scaler = torch.load(model_path / 'prop_scaler.pt')

        if load_data:
            datamodule = hydra.utils.instantiate(
                cfg.data.datamodule, _recursive_=False, scaler_path=model_path
            )
            if testing:
                datamodule.setup('test')
                test_loader = datamodule.test_dataloader()[0]
            else:
                datamodule.setup()
                test_loader = datamodule.val_dataloader()[0]
        else:
            test_loader = None

    return model, test_loader, cfg


def get_target_cryst(model, ld_kwargs, data_loader,
                 num_points=256, num_gradient_steps=5000,
                 lr=1e-3, num_saved_crys=4,seed=42):
    
    
    if data_loader is not None:
        batch = next(iter(data_loader)).to(model.device)


        torch.manual_seed(seed)
        z = torch.randn(model.hparams.latent_dim, model.hparams.hidden_dim,
                       device=model.device) 
       
        
        z = z[:batch.size(0)].detach().clone()
        #print(z.shape)
        z.requires_grad = True
        all_target_nums = batch.num_atoms
        all_target_atoms = batch.atom_types
        all_batch = batch.batch
        
        
    else:
        z = torch.randn(num_points, model.hparams.hidden_dim,
                        device=model.device)
        z.requires_grad = True


        with open(f'{os.environ["HYDRA_JOBS"]}/target_atoms.txt') as f:
            lines = f.readlines()
            lines = [line.strip().replace("'","\"") for line in lines]
        target_dict = json.loads(lines[0])

        target_nums = sum(target_dict.values())
        target_atoms = []
        for key, value in target_dict.items():
            target_atoms += [chemical_symbols.index(key) for _ in range(value)]

        all_target_nums = [target_nums for _ in range(num_points)]
        all_target_atoms, all_batch = [], []
        for i in range(num_points):
            all_target_atoms += target_atoms
            all_batch += [i for _ in range(target_nums)]

        all_target_nums=torch.tensor(all_target_nums).to(model.device)
        all_target_atoms=torch.tensor(all_target_atoms).to(model.device)
        all_batch=torch.tensor(all_batch).to(model.device)    
    
    opt = Adam([z], lr=lr)
    model.freeze()

    all_crystals = []
    all_energys = []
    all_z = []
    
    def get_composition_loss( pred_composition_per_atom, target_atom_types, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction='none')
        return scatter(loss, batch, reduce='mean').mean()
    
    

    interval = num_gradient_steps // (num_saved_crys-1)
    for i in tqdm(range(num_gradient_steps)):
        opt.zero_grad()

        ## sh ##
        
        num_atoms = model.predict_num_atoms(z)
        loss_num = F.cross_entropy(num_atoms, all_target_nums)
        composition_per_atom = model.predict_composition(z, all_target_nums)
      
        composition_loss = get_composition_loss(
            composition_per_atom, all_target_atoms, all_batch)
        

        loss = (0.5*loss_num + 0.5* composition_loss)

        loss.backward()
        opt.step()

        if i ==0 :
            continue
        if i%interval ==0 or i == (num_gradient_steps-1):
            crystals = model.langevin_dynamics(z, ld_kwargs)

            prop=model.scaler.inverse_transform(model.fc_property(z))
            all_crystals.append(crystals)
            all_energys.append(prop)
            all_z.append(z)

        
            
    all_energys = torch.cat(all_energys)
    all_z = torch.cat(all_z)
    result_dic = ({k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0) for k in ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']})
        
    result_dic['energy'] = all_energys
    result_dic['z'] = all_z
    #result_dic['initial_type'] = torch.cat([d['initial_type'] for d in all_crystals]).unsqueeze(0)
    return result_dic




        
def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(
        model_path, load_data=False)
         
        
    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar)

    if torch.cuda.is_available():
        model.to('cuda')
    




    start_time = time.time()
    if args.start_from == 'data':
        loader = test_loader
    else:
        loader = None
    optimized_crystals = get_target_cryst(model, ld_kwargs, loader, num_gradient_steps=args.num_gradient,
                                     num_saved_crys= args.num_crys, seed=args.seed)
    optimized_crystals.update({'eval_setting': args,
                               'time': time.time() - start_time})

    if args.label == '':
        gen_out_name = 'gen_target.pt'
    else:
        gen_out_name = f'gen_target_{args.label}.pt'
    torch.save(optimized_crystals, model_path / gen_out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)

    parser.add_argument('--n_step_each', default=100, type=int)
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--start_from', default='data', type=str)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--force_num_atoms', action='store_true')
    parser.add_argument('--force_atom_types', action='store_true')
    parser.add_argument('--down_sample_traj_step', default=10, type=int)
    parser.add_argument('--label', default='')
    parser.add_argument('--num_gradient', default=5000, type=int)
    parser.add_argument('--num_crys', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    main(args)

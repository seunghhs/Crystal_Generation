import torch
import pandas as pd
from tqdm import tqdm
from pymatgen.analysis.structure_matcher import StructureMatcher
from collections import defaultdict
import copy
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from cdvae.common.data_utils import chemical_symbols


def get_cryst_info(data_dict, i, ):
    new_dict = {}
    nums = len(data_dict['lengths'][0])
    new_dict['lengths'] = data_dict['lengths'][0][i].tolist()
    new_dict['angles'] = data_dict['angles'][0][i].tolist()
    
    atom_nums= data_dict['num_atoms'][0]
    start = sum(atom_nums[:i])
    end = start + atom_nums[i]
    
    new_dict['num_atoms'] = atom_nums[i].item()
    new_dict['atom_types'] = data_dict['atom_types'][0][start:end].tolist()
    new_dict['frac_coords'] = data_dict['frac_coords'][0][start:end].tolist()
    
    return new_dict

def make_structure(new_dict, chemical_symbols=chemical_symbols):
    lengths = new_dict['lengths']
    angles = new_dict['angles']
    coords = new_dict['frac_coords']
    lattice = Lattice.from_parameters(a=lengths[0], b= lengths[1], c=lengths[2],
                                     alpha= angles[0], beta=angles[1], gamma=angles[2])
    
    atom_types = [chemical_symbols[i] for i in new_dict['atom_types']]
    
    struct = Structure(lattice, atom_types, coords)
    
    return struct


def get_all_structures( gen_ckpt_file,  energy = False):

    structures=[]
    energies =[]
    generated =torch.load(gen_ckpt_file)
    nums = len(generated['lengths'][0])
    for i in tqdm(range(nums)):
        
        generated_dic=get_cryst_info(generated,i)
        struct=make_structure(generated_dic)
        structures.append(struct)
        if energy:
            e = generated['energy'][i].item()
            energies.append(e)
            
    if energy:
        return structures,  energies
    return structures


def get_cif_from_structure(structures):
    from pymatgen.io.cif import CifWriter
    cif_lines, energy_lines = [],[]
    
    for i, struct in enumerate(structures):
        cif = CifWriter(struct, )
        cif_lines.append(cif.__str__())
    return cif_lines

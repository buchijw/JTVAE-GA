import torch
from pathlib import Path
import sys, os
import argparse

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
import networkx as nx
from tqdm import tqdm

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from buchiutils.train_utils import save_train_hyper_csv
sys.path.append('%s/../fast_jtnn/' % os.path.dirname(os.path.realpath(__file__)))
from mol_tree import Vocab, MolTree
from jtprop_vae import JTPropVAE
import numpy as np
import pandas as pd

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', type=str, required=True)
parser.add_argument('--smiles_file', type=str, required=True)
parser.add_argument('--ID_file', type=str,default='')
parser.add_argument('--model',type=str, required=True)
parser.add_argument('--save_dir',type=str, required=True)
parser.add_argument('--sim', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=2.0)
parser.add_argument('--lr_list_file', type=str, default='')
parser.add_argument('--n_iter', type=int, default=80)
parser.add_argument('--seed', type=int, default=42)

# Model configurations
parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

args = parser.parse_args()
hyper_dict = vars(args)

torch.manual_seed(args.seed)

# Load vocab
vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab_set = set(vocab)
vocab = Vocab(vocab)

def check_vocab(smiles):
    """
    Check if the given smiles string is a valid vocabulary by comparing each node in the molecular tree with the vocabulary set.

    Parameters:
        smiles (str): The smiles string to be checked.

    Returns:
        bool: True if all nodes in the molecular tree are present in the vocabulary set, False otherwise.
    """
    cset = set()
    mol = MolTree(smiles)
    for c in mol.nodes:
        cset.add(c.smiles)
    return cset.issubset(vocab_set)

# Load the model
model = JTPropVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG).cuda()
model.load_state_dict(torch.load(args.model))

sim_cutoff = float(args.sim)
lr = float(args.lr)
n_iter = int(args.n_iter)

with open(args.smiles_file) as f:
    data = [line.rstrip('\n') for line in f]

if args.ID_file not in ['','None']:
    with open(args.ID_file) as f:
        ID_list = [line.rstrip('\n') for line in f]
    assert len(ID_list)==len(data)
else:
    ID_list=list(range(len(data)))

if args.lr_list_file != '':
    with open(args.lr_list_file) as f:
        lr_list = [float(line.rstrip('\n')) for line in f]
else:
    lr_list = [lr]

df = pd.DataFrame(columns=['id','smiles','run','from'])

# res = []
Path.mkdir(Path(args.save_dir),exist_ok=True)
save_train_hyper_csv(hyper_dict,args.save_dir+'/settings.csv')


for lr in lr_list:
    print('RUNNING LR = %.2f'%(lr))
    for idx,smiles in tqdm(enumerate(data),total=len(data)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print('SMILES at index %d is invalid.' % (idx))
            continue
        if not check_vocab(smiles):
            print('SMILES at index %d is not in vocab.' % (idx))
            continue

        result = model.optimize_grad_ascent(smiles, sim_cutoff=sim_cutoff, lr=lr, num_iter=n_iter)
        
        assert len(result)==n_iter
        
        print('-- Result for idx: %s --'%(str(ID_list[idx])))
        print('Failed: %d / %d'%(result.count('None'),n_iter))
        diff_smiles = set(result)
        if 'None' in diff_smiles:
            diff_smiles.remove('None')
        if smiles in diff_smiles:
            diff_smiles.remove(smiles)
        print('New SMILES: %d / %d'%(len(diff_smiles),n_iter))
        
        Path.mkdir(Path(args.save_dir + '/%s'%(str(ID_list[idx]))),exist_ok=True)
        with open(args.save_dir + '/%s'%(str(ID_list[idx])) + '/%s-%.2f-valid.txt'%(str(ID_list[idx]),lr),'w') as f:
            txt = '\n'.join(diff_smiles)+'\n'
            f.write(txt)
        for i,smi in enumerate(diff_smiles):
            if smi not in df.smiles.values:
                new_row = {
                    'id':['GA%d'%(df.shape[0])],
                    'smiles':[smi],
                    'run':['%s_%.2f'%(str(ID_list[idx]),lr)],
                    'from':['%s'%(str(ID_list[idx]))]
                    }
                df = pd.concat([df,pd.DataFrame(new_row)])
                df.reset_index(drop=True,inplace=True)
df.to_csv(args.save_dir+'/result.csv',index=False)


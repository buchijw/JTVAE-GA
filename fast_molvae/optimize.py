import torch
# import torch.nn as nn
# from torch.autograd import Variable

import sys
import argparse
from collections import deque

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
# import sascorer
import networkx as nx
from tqdm import tqdm

import os
sys.path.append('%s/../fast_jtnn/' % os.path.dirname(os.path.realpath(__file__)))
from mol_tree import Vocab, MolTree
from jtprop_vae import JTPropVAE
# from jtnn_enc import JTNNEncoder
# from jtmpn import JTMPN
# from mpn import MPN
# from nnutils import create_var
# from datautils import PropMolTreeFolder, PairTreeFolder, PropMolTreeDataset

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
# parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--test', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--sim', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=2.0)
parser.add_argument('--n_iter', type=int, default=80)
# parser.add_argument('--save_dir', required=True)
# parser.add_argument('--load_epoch', type=int, default=0)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)


# parser.add_argument('--clip_norm', type=float, default=50.0)
# parser.add_argument('--beta', type=float, default=0.0)
# parser.add_argument('--step_beta', type=float, default=0.002)
# parser.add_argument('--max_beta', type=float, default=1.0)
# parser.add_argument('--warmup', type=int, default=40000)

# parser.add_argument('--epoch', type=int, default=20)
# parser.add_argument('--anneal_rate', type=float, default=0.9)
# parser.add_argument('--anneal_iter', type=int, default=40000)
# parser.add_argument('--kl_anneal_iter', type=int, default=2000)
# parser.add_argument('--print_iter', type=int, default=50)
# parser.add_argument('--save_iter', type=int, default=5000)

args = parser.parse_args()

def penalized_logp_standard(mol):

    logP_mean = 2.4399606244103639873799239
    logP_std = 0.9293197802518905481505840
    SA_mean = -2.4485512208785431553792478
    SA_std = 0.4603110476923852334429910
    cycle_mean = -0.0307270378623088931402396
    cycle_std = 0.2163675785228087178335699

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length
    # print(logP_mean)

    standardized_log_p = (log_p - logP_mean) / logP_std
    standardized_SA = (SA - SA_mean) / SA_std
    standardized_cycle = (cycle_score - cycle_mean) / cycle_std
    return standardized_log_p + standardized_SA + standardized_cycle
   
vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

model = JTPropVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG).cuda()

# hidden_size = int(opts.hidden_size)
# latent_size = int(opts.latent_size)
# depth = int(opts.depth)
sim_cutoff = float(args.sim)
lr = float(args.lr)
n_iter = int(args.n_iter)

model.load_state_dict(torch.load(args.model))

# data = []
with open(args.test) as f:
    data = [line.rstrip('\n') for line in f]

res = []
for idx,smiles in tqdm(enumerate(data)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('SMILES at index %d is invalid.' % (idx))
        continue
    score = penalized_logp_standard(mol)

    new_smiles,sim = model.optimize(smiles, sim_cutoff=sim_cutoff, lr=lr, num_iter=n_iter)
    new_mol = Chem.MolFromSmiles(new_smiles)
    if new_mol is None:
        print('New SMILES at index %d is invalid.' % (idx))
        continue
    new_score = penalized_logp_standard(new_mol)

    res.append( (new_score - score, sim, score, new_score, smiles, new_smiles) )
    print(new_score - score, sim, score, new_score, smiles, new_smiles)

print('Sum increased score: %.5f \t Sum sim: %.5f' % (sum([x[0] for x in res]), sum([x[1] for x in res])))

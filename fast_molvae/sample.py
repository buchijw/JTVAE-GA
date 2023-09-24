import torch
import torch.nn as nn

import argparse

import os
import sys
sys.path.append('%s/../fast_jtnn/' % os.path.dirname(os.path.realpath(__file__)))
from mol_tree import Vocab, MolTree
from jtnn_vae import JTNNVAE, JTPropVAE
from jtnn_enc import JTNNEncoder
from jtmpn import JTMPN
from mpn import MPN
from nnutils import create_var
from datautils import MolTreeFolder, PairTreeFolder, MolTreeDataset, PropMolTreeFolder, PropMolTreeDataset
# from fast_jtnn import *
import rdkit
from tqdm import tqdm

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--nsample', type=int, required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--prop', type=bool, default=False)
parser.add_argument('--seed', required=True,default=2023)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)
parser.add_argument('--result_file', type=str, required=True)

args = parser.parse_args()
   
vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

if not args.prop:
    model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG)
else:
    model = JTPropVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG)
model.load_state_dict(torch.load(args.model))
model = model.cuda()

torch.manual_seed(args.seed)
result =[]
for i in tqdm(range(args.nsample)):
    sam  = model.sample_prior()
    if sam is not None:
        result.append(sam)
        
with open(args.result_file, 'w') as f:
    for smi in result:
        f.write(f"{smi}\n")

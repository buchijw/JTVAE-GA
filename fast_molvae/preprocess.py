import torch
import torch.nn as nn
from joblib import Parallel,delayed
import rdkit
from rdkit import Chem
import argparse
import pickle
from tqdm import tqdm

import os
import sys
sys.path.append('%s/../fast_jtnn/' % os.path.dirname(os.path.realpath(__file__)))
from mol_tree import Vocab, MolTree
from jtnn_vae import JTNNVAE
from jtnn_enc import JTNNEncoder
from jtmpn import JTMPN
from mpn import MPN
from nnutils import create_var
from datautils import MolTreeFolder, PairTreeFolder, MolTreeDataset
# from fast_jtnn import *
from proputils import penalized_logp_standard


def tensorize(smiles, assm=True):
    """
    Converts a SMILES representation of a molecule into a MolTree object.

    Args:
        smiles (str): The SMILES representation of the molecule.
        assm (bool, optional): Whether to assemble the molecule tree. Defaults to True.

    Returns:
        MolTree: The MolTree object representing the molecule.
    """
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree

def tensorize_prop(smiles, assm=True):
    """
    Generates a molecular tree representation of a given SMILES string and calculates the penalized 
    logP standard property of the molecule.

    Parameters:
    - smiles (str): The SMILES string representing the molecule.
    - assm (bool): A flag indicating whether to perform assembly of the molecular tree. 
                   Default is True.

    Returns:
    - mol_tree (MolTree): The molecular tree representation of the molecule.
    - prop (float): The penalized logP standard property of the molecule.
    """
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol
    
    prop = penalized_logp_standard(Chem.MolFromSmiles(smiles))

    return (mol_tree, prop)

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", dest="train_path")
    parser.add_argument("-p", "--prop", dest="prop",type=bool,default=False)
    parser.add_argument("-n", "--split", dest="nsplits", type=int, default=10)
    parser.add_argument("-j", "--ncpu", dest="ncpu", type=int, default=8)
    args = parser.parse_args()
    args.ncpu = int(args.ncpu)

    num_splits = int(args.nsplits)

    with open(args.train_path) as f:
        smiles_list = [line.rstrip("\n") for line in f]
    
    if args.prop:
        all_data = list(Parallel(n_jobs=args.ncpu, backend="loky")(
            delayed(tensorize_prop)(smiles_list[idx],True) for idx in tqdm(range(0,len(smiles_list)))
                )
            )
    else:
        all_data = list(Parallel(n_jobs=args.ncpu, backend="loky")(
            delayed(tensorize)(smiles_list[idx],True) for idx in tqdm(range(0,len(smiles_list)))
                )
            )
    
    le = int((len(all_data) + num_splits - 1) / num_splits)

    for split_id in range(num_splits):
        st = split_id * le
        sub_data = all_data[st : st + le]

        with open('tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)


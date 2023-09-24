'''
Usage:
--get_smiles $smiles_set --output $output_txt
    Get SMILES from set $smiles_set: moses,... and save to $output_txt
--smiles_file $smiles_path --output $output_csv --ncpu $ncpu
    Calculate properties (SMILES, MW, LogP, TPSA, HBD, HBA) and save to $output_csv using $ncpu cpu(s).
'''
import os
import argparse
import moses
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcNumHBD
from rdkit.Chem.rdMolDescriptors import CalcNumHBA
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit import Chem
import multiprocessing
from multiprocessing import Pool

def cal_prop(s):
    m = Chem.MolFromSmiles(s)
    if m is None : return None
    # return Chem.MolToSmiles(m), ExactMolWt(m), MolLogP(m), CalcTPSA(m)
    return Chem.MolToSmiles(m), ExactMolWt(m), MolLogP(m), CalcTPSA(m), CalcNumHBD(m), CalcNumHBA(m)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--get_smiles', help='choose sets: moses,...', type=str, default='0')
    parser.add_argument('--output', help='output file path', type=str, default='data_train.txt')
    parser.add_argument('--ncpu', help='number of cpu for properties calculation', type=int, default=1)
    parser.add_argument('--smiles_file', help='smiles file path for properties calculation', type=str, default='0')
    args = parser.parse_args()
    
    if args.get_smiles == 'moses':
        print('Get train SMILES data from set: ',args.get_smiles)
        print('Write SMILES set to file: ',args.output)
        train = moses.get_dataset('train')
        print('Total: ',len(train))
        with open(args.output, 'w') as f:
            for smiles in train:
                f.write(f"{smiles}\n")

    if args.smiles_file != '0':
        print('Calculate properties (MW, LogP, TPSA, HBD, HBA) from SMILES file: ',args.smiles_file)
        print('Write (SMILES, MW, LogP, TPSA, HBD, HBA) to file: ',args.output)
        #Lỗi của MacOS
        multiprocessing.set_start_method("fork")
        with open(args.smiles_file,'r') as f:
            smiles_list = [line.rstrip('\n') for line in f]
        pool = Pool(args.ncpu)
        
        r = pool.map_async(cal_prop, smiles_list)
        
        data = r.get()
        pool.close()
        pool.join()
        i = 0
        with open(args.output, 'w') as f:
            for d in data:
                if d is None:
                    continue
                f.write(d[0] + '\t' + str(d[1]) + '\t'+ str(d[2]) + '\t'+ str(d[3]) + '\t'+ str(d[4]) + '\t'+ str(d[5]) + '\n')
                i += 1
        print('Total: ',i)
    
    print('DONE!')
'''
Usage:
--smiles_file $smiles_path
    Get metrics from SMILES file $smiles_path (min 30,000 SMILES)
--smiles_file $smiles_path --output $output_pkl
    Get metrics from SMILES file $smiles_path (min 30,000 SMILES) and save metrics dict to $output_pkl
--metrics_file $metrics_pkl
    Read and print metrics dict from $metrics_pkl
'''

import os
import argparse
import moses
import pickle

def get_metrics(smiles_file):
    with open(smiles_file,'r') as f:
        smiles_list = [line.rstrip('\n') for line in f]
    metrics = moses.get_all_metrics(smiles_list, n_jobs = args.n_cpus)
    return metrics

def read_metrics(result_file):
    with open(result_file, 'rb') as f:
        metrics = pickle.load(f)
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles_file', help='sample smiles file path', type=str, default='0')
    parser.add_argument('--output', help='output file path if want to save result', type=str, default='0')
    parser.add_argument('--metrics_file', help='metrics file path to read', type=str, default='0')
    parser.add_argument('--n_cpus', help='number of cpus', type=int, default=1)
    args = parser.parse_args()
    
    if args.smiles_file != '0':
        print('Get all metrics using SMILES file: ',args.smiles_file)
        metrics = get_metrics(args.smiles_file)
        for key, value in metrics.items():
            print(key, ' : ', value)
    
        if args.output != '0':
            print('Write result to file: ',args.output)
            with open(args.output,'wb') as f:
                pickle.dump(metrics,f)
    
    if args.metrics_file != '0':
        metrics = read_metrics(args.metrics_file)
        for key, value in metrics.items():
            print(key, ' : ', value)
            
    print('DONE!')
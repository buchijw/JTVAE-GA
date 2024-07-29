#!/bin/bash
vocab=../moses/vocab.txt
model=../fast_molvae/moses_h450z56_prop/model.iter-730000

smiles_file=./VEGFR2_smiles.txt
ID_file=./VEGFR2_smiles_name.txt
save_dir=./VEGFR2

sim=0.3
# lr is ignored if lr_list_file is set
lr=0.03
lr_list_file=./lr_list.txt
n_iter=50

seed=42
python ga.py \
    --vocab $vocab \
    --model $model \
    --smiles_file $smiles_file \
    --ID_file $ID_file \
    --save_dir $save_dir \
    --sim $sim \
    --lr $lr \
    --n_iter $n_iter \
    --seed $seed \
    --lr_list_file $lr_list_file
# Training JTVAE **without** properties

## Set up environment variables

Suppose the repository is downloaded at `$PREFIX/JTVAE-GA` directory. First set up environment variables:

```sh
export PYTHONPATH=$PREFIX/JTVAE-GA
```

## Dataset

The MOSES dataset is in `JTVAE-GA/moses` (derived from [https://github.com/molecularsets/moses](https://github.com/molecularsets/moses)). The vocabulary is included.

If you use a new dataset, please derive the new vocabulary from your dataset.

## Deriving Vocabulary

If you are running on a new dataset, you need to compute the vocabulary from your dataset.
To perform tree decomposition over a set of molecules, run

```sh
python ../fast_jtnn/mol_tree.py --smiles_path train.txt --vocab_path vocab.txt --ncpu 8
```

This gives you the vocabulary of cluster labels over the dataset `train.txt`.

Replace `train.txt` to your SMILES file's path and `vocab.txt` to the vocabulary destination. Default value of `--ncpu` is `8`.

## Training

### Step 1: Preprocess the data:

```sh
python preprocess.py --train train.txt --split 100 --jobs 16
mkdir moses-processed
mv tensor* moses-processed
```

This script will preprocess the training data (subgraph enumeration & tree decomposition), and save results into a list of files. We suggest you to use small value for `--split` if you are working with smaller datasets. `--jobs 16` means to use 16 CPUs.

### Step 2: Train VAE model with KL annealing.

```sh
mkdir vae_model/
python vae_train.py --train moses-processed --vocab ../moses/vocab.txt --save_dir vae_model/
```

Default Options:

`--beta 0` means to set KL regularization weight (beta) initially to be zero.

`--warmup 40000` means that beta will not increase within first 40000 training steps. It is recommended because using large KL regularization (large beta) in the beginning of training is harmful for model performance.

`--step_beta 0.002 --kl_anneal_iter 1000` means beta will increase by 0.002 every 1000 training steps (batch updates). You should observe that the KL will decrease as beta increases.

`--max_beta 1.0 ` sets the maximum value of beta to be 1.0.

`--save_dir vae_model`: the model will be saved in vae_model/

Please note that this is not necessarily the best annealing strategy. You are welcomed to adjust these parameters.

Hyperparameters and progress are recorded in two `.csv` files in the same `save_dir` folder with the model.

## Testing

To sample new molecules with trained models, simply run

```sh
python sample.py --nsample 100000 --vocab ../moses/vocab.txt --model model.iter-xxxx --result_file sample.txt --seed 2023
```

This script samples 100K SMILES strings with seed `2023`. Replace `model.iter-xxxx` with a path to your model checkpoint.
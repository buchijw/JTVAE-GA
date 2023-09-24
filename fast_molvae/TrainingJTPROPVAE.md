# Training JTVAE **with** properties

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
python preprocess.py --train train.txt --split 100 --jobs 16 --prop True 
mkdir moses-processed
mv tensor* moses-processed
```

This script will preprocess the training data (subgraph enumeration & tree decomposition), and save results into a list of files. We suggest you to use small value for `--split` if you are working with smaller datasets. `--jobs 16` means to use 16 CPUs. `--prop` is set to `True` to calculate penalized logP values for all molecules. 

If you want to use other property, please take a look at the `tensorize_prop` function in `preprocess.py`.

### Step 2: Train VAE model with KL annealing.

```sh
mkdir vae_model/
python vae_train_prop.py --train moses-processed --vocab ../moses/vocab.txt --save_dir vae_model/
```

Default Options:

`--beta 0` means to set KL regularization weight (beta) initially to be zero.

`--warmup 40000` means that beta will not increase within first 40000 training steps. It is recommended because using large KL regularization (large beta) in the beginning of training is harmful for model performance.

`--step_beta 0.002 --kl_anneal_iter 1000` means beta will increase by 0.002 every 1000 training steps (batch updates). You should observe that the KL will decrease as beta increases.

`--max_beta 1.0 ` sets the maximum value of beta to be 1.0.

`--save_dir vae_model`: the model will be saved in vae_model/

Please note that this is not necessarily the best annealing strategy. You are welcomed to adjust these parameters.

Hyperparameters and progress are recorded in two `.csv` files in the same `save_dir` folder with the model.

### Train progress tracking

We added further train progress tracking with `Tensorboard` and `wandb`. Please use the following alternatives instead of `vae_train_prop.py` in training:

* `vae_train_prop_tensorboard.py` for tracking with `TensorBoard`
* `vae_train_prop_wandb.py` for tracking with `wandb`

## Testing

To sample new molecules with trained models, simply run

```sh
python sample.py --nsample 100000 --vocab ../moses/vocab.txt --model moses_h450z56_prop/model.iter-730000 --result_file sample.txt --seed 2023 --prop True
```

This script samples 100K SMILES strings with seed `2023`. Replace `moses_h450z56_prop/model.iter-730000` with a path to your model checkpoint.

`--prop` is set to `True` to use JTVAE with properties. 

The `moses_h450z56_prop/model.iter-730000` file is a model trained with 730K steps with the following hyperparameters:

```ini
hidden_size = 450
latent_size = 56
batch_size = 64
depthT = 20
depthG = 3
lr = 0.0007
clip_norm = 50.0
beta = 0.0
step_beta = 0.002
max_beta = 0.1
warmup = 60000
epoch = 50
anneal_rate = 0.9
anneal_iter = 40000
kl_anneal_iter = 2000
print_iter = 50
save_iter = 5000
```

The MOSES evaluation result of this model is as follows::

```ini
valid = 1.0
unique@1000 = 1.0
unique@10000 = 0.9998 ± 0.0002
FCD/Test = 0.9049 ± 0.0060
SNN/Test = 0.5164 ± 0.0005
Frag/Test = 0.9914 ± 0.0001
Scaf/Test = 0.8561 ± 0.0037
FCD/TestSF = 1.6347 ± 0.0119
SNN/TestSF = 0.4953 ± 0.0005
Frag/TestSF = 0.9882 ± 0.0001
Scaf/TestSF = 0.1708 ± 0.0029
IntDiv = 0.8517 ± 0.0001
IntDiv2 = 0.8462 ± 0.0001
Filters = 0.9653 ± 0.0003
logP = 0.2268 ± 0.0026
SA = 0.1545 ± 0.0029
QED = 0.0044 ± 0.0001
weight = 12.6952 ± 0.1238
Novelty = 0.9671 ± 0.0005
```
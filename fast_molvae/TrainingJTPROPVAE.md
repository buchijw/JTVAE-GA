<a name="readme-top"></a>

# Training JTVAE **with** properties

<!-- TABLE OF CONTENTS -->
<details open>
  <summary><h3>Table of Contents</h3></summary>
  <ol>
    <li><a href="#set-up-environment-variables">Set up environment variables</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#deriving-vocabulary">Deriving Vocabulary</a></li>
    <li>
        <a href="#training">Training</a>
        <ul>
            <li><a href="#step-1-preprocess-the-data">Step 1: Preprocess the data</a></li>
            <li><a href="#step-2-train-vae-model-with-kl-annealing">Step 2: Train VAE model with KL annealing</a></li>
            <li><a href="#train-progress-tracking">Train progress tracking</a></li>
        </ul>
    </li>
    <li><a href="#testing">Testing</a></li>
    <li><a href="#pre-trained-model">Pre-trained model</a></li>
  </ol>
</details>

## Set up environment variables

Suppose the repository is downloaded at `$PREFIX/JTVAE-GA` directory. First set up environment variables:

```sh
export PYTHONPATH=$PREFIX/JTVAE-GA
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Dataset

The MOSES dataset is in `JTVAE-GA/moses` (derived from [https://github.com/molecularsets/moses](https://github.com/molecularsets/moses)). The vocabulary is included.

If you use a new dataset, please derive the new vocabulary from your dataset.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Deriving Vocabulary

If you are running on a new dataset, you need to compute the vocabulary from your dataset. Please make sure your SMILES strings are valid. Stereo generation is disabled by default.

To perform tree decomposition over a set of molecules, run:

```sh
python ../fast_jtnn/mol_tree.py --smiles_path train.txt --vocab_path vocab.txt --ncpu 8
```

This gives you the vocabulary of cluster labels over the dataset `train.txt`.

| Parameter | Type     | Default | Description                |
| :-------- | :------- | :------ | :------------------------- |
| `--smiles_path` | `str` | | **Required**. Path to your SMILES file. |
| `--vocab_path` | `str` | | **Required**. Path to the vocabulary destination. |
| `--ncpu` | `int` | `8` | Number of CPUs to use. |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Training

### Step 1: Preprocess the data

```sh
python preprocess.py --train train.txt --split 100 --ncpu 16 --prop True 
mkdir moses-processed
mv tensor* moses-processed
```

This script will preprocess the training data (subgraph enumeration & tree decomposition), and save results into a list of files. We suggest you to use small value for `--split` if you are working with smaller datasets. `--prop` is set to `True` to calculate penalized logP values for all molecules.

| Parameter | Type     | Default | Description                |
| :-------- | :------- | :------ | :------------------------- |
| `--train` | `str` | | **Required**. Path to your training SMILES file. |
| `--split` | `str` | `10` | Number of split result files. |
| `--ncpu` | `int` | `8` | Number of CPUs to use. |
| `--prop` | `bool` | `False` | Calculate properties. |

If you want to use other property, please take a look at the `tensorize_prop` function in `preprocess.py`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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

| Parameter | Type     | Default | Description                |
| :-------- | :------- | :------ | :------------------------- |
| `--train` | `str` | | **Required**. Path to directory that contains preprocessed data. |
| `--vocab` | `str` | | **Required**. Path to the vocabulary file. |
| `--save_dir` | `str` | | **Required**. Path to directory that contains model checkpoints. |
| `--load_epoch` | `int` | `0` | Model checkpoint to load. `0` means no model checkpoint. |
| `--hidden_size` | `int` | `450` | Hidden layer size. |
| `--batch_size` | `int` | `32` | Batch size. |
| `--latent_size` | `int` | `56` | Latent size. |
| `--depthT` | `int` | `20` | Depth of tree NN. |
| `--depthG` | `int` | `3` | Depth of graph NN. |
| `--lr` | `float` | `0.001` | Initial learning rate. |
| `--clip_norm` | `float` | `50.0` | Clip norm. |
| `--beta` | `float` | `0.0` | Initial beta. |
| `--step_beta` | `float` | `0.002` | Beta inscreasing step. |
| `--max_beta` | `float` | `1.0` | Max beta. |
| `--warmup` | `int` | `40000` | Number of first training steps with no beta inscreasing. |
| `--epoch` | `int` | `20` | Number of epochs. |
| `--anneal_rate` | `float` | `0.9` | Decay rate. |
| `--anneal_iter` | `int` | `40000` | Number of training steps needed for learning rate decay. |
| `--kl_anneal_iter` | `int` | `2000` | Number of training steps needed for beta increasing. |
| `--print_iter` | `int` | `50` | Number of training steps needed to print training progress. |
| `--save_iter` | `int` | `5000` | Number of training steps needed for a checkpoint. |

Hyperparameters and progress are recorded in two `.csv` files in the same `save_dir` folder with the model.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Train progress tracking

We added further train progress tracking with `Tensorboard` and `wandb`. Please use the following alternatives instead of `vae_train_prop.py` in training:

* `vae_train_prop_tensorboard.py` for tracking with `TensorBoard`
* `vae_train_prop_wandb.py` for tracking with `wandb`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Testing

To sample new molecules with trained models, simply run

```sh
python sample.py --nsample 100000 --vocab ../moses/vocab.txt --model moses_h450z56_prop/model.iter-730000 --result_file sample.txt --seed 2023 --prop True
```

This script samples 100K SMILES strings with seed `2023`. Replace `moses_h450z56_prop/model.iter-730000` with a path to your model checkpoint.

`--prop` is set to `True` to use JTVAE with properties.

| Parameter | Type     | Default | Description                |
| :-------- | :------- | :------ | :------------------------- |
| `--nsample` | `int` | | **Required**. Number of samples. |
| `--vocab` | `str` | | **Required**. Path to the vocabulary file. |
| `--model` | `str` | | **Required**. Path to a model checkpoint. |
| `--result_file` | `str` | | **Required**. Path to the result file. |
| `--prop` | `bool` | `False` | Whether to use JTPROPVAE or not. |
| `--seed` | `int` | `2023` | Random seed. |
| `--hidden_size` | `int` | `450` | Hidden layer size. |
| `--latent_size` | `int` | `56` | Latent size. |
| `--depthT` | `int` | `20` | Depth of tree NN. |
| `--depthG` | `int` | `3` | Depth of graph NN. |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Pre-trained model

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

<p align="right">(<a href="#readme-top">back to top</a>)</p>
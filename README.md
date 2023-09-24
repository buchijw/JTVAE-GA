# Gradient Ascent with Junction Tree Variational Autoencoder (JTVAE-GA)

Gradient Ascent (GA) was implemented using Junction Tree Variational Autoencoder (JTVAE) based on the accelerated version on [the official repo](https://github.com/wengong-jin/icml18-jtnn). Penalized logP was chosen to be the target molecular property.

Official Junction Tree Variational Autoencoder belongs to Wengong Jin (wengong@csail.mit.edu), Regina Barzilay, Tommi Jaakkola [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364).

# Requirements

This git repo requires Git LFS installed for big datasets and model checkpoints. To clone this repo, please run:

```sh
git lfs install
git clone https://github.com/buchijw/JTVAE-GA.git
```

We tested on Linux (Ubuntu 22.04 LTS with ROCm 5.4.2) and MacOS (Ventura 13.5 with Intel CPU), therefore the code supposes to work on CUDA/HIP/CPU based on type of PyTorch installed.

Packages (versions in brackets were used):

* `Python` (3.10.9)
* `RDKit` (2022.03.2)
* `PyTorch` (2.0.1 with ROCm 5.4.2)
* `tqdm` (4.65.0)
* `networkx` (3.0)
* `joblib` (1.2.0)
* `pandas` (1.5.3)
* `numpy` (1.24.3)
* `wandb` (0.15.4): optional, used for tracking training progress with wandb.
* `tensorboard`: optional, used for tracking training progress with TensorBoard.

# Quick Start

The following directories contains the implementations of the model:

* `fast_jtnn/` contains codes for model implementation.
* `fast_molvae/` contains codes for VAE training. Please refer to `fast_molvae/README.md` for details.

The MOSES datasets were taken from https://github.com/molecularsets/moses. Please refer to `moses/README.md` for details.

# Contact

Gia-Bao Truong (tgbao.d18@ump.edu.vn)

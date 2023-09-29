<a name="readme-top"></a>

# Gradient Ascent with Junction Tree Variational Autoencoder (JTVAE-GA)

![GitHub issues](https://img.shields.io/github/issues/buchijw/JTVAE-GA?style=for-the-badge)
![GitHub pull requests](https://img.shields.io/github/issues-pr/buchijw/JTVAE-GA?style=for-the-badge)
![License](https://img.shields.io/github/license/buchijw/JTVAE-GA?style=for-the-badge)
![Git LFS](https://img.shields.io/badge/GIT%20LFS-8A2BE2?style=for-the-badge)

<!-- TABLE OF CONTENTS -->
<details open>
  <summary><h3>Table of Contents</h3></summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#requirements">Requirements</a></li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
Gradient Ascent (GA) was implemented using Junction Tree Variational Autoencoder (JTVAE) based on the accelerated version on [the official repo](https://github.com/wengong-jin/icml18-jtnn). Penalized logP was chosen to be the target molecular property.

Official Junction Tree Variational Autoencoder belongs to Wengong Jin (wengong@csail.mit.edu), Regina Barzilay, Tommi Jaakkola [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- REQUIREMENTS -->
## Requirements

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

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- QUICK START -->
## Quick Start

The following directories contains the implementations of the model:

* `fast_jtnn/` contains codes for model implementation.
* `fast_molvae/` contains codes for VAE training. Please refer to `fast_molvae/README.md` for details.

The MOSES datasets were taken from https://github.com/molecularsets/moses. Please refer to `moses/README.md` for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Gia-Bao Truong (tgbao.d18@ump.edu.vn)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Wengong Jin, Regina Barzilay and Tommi Jaakkola for [Junction Tree Variational Autoencoder](https://arxiv.org/abs/1802.04364).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

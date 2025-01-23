# ARTEMIS

## Abstract
Cellular processes like development, differentiation, and disease progression are highly complex and dynamic (e.g., gene expression). These processes often undergo cell population changes driven by cell birth, proliferation, and death. Single-cell sequencing enables gene expression measurement at single-cell resolution, allowing us to decipher cellular and molecular dynamics underlying these processes. However, the high costs and destructive nature of sequencing restrict observations to snapshots of unaligned cells at discrete timepoints, limiting our understanding of these processes and complicating the reconstruction of cellular trajectories.
To address this challenge, we propose ARTEMIS, a generative model integrating a variational autoencoder (VAE) with unbalanced diffusion schrödinger bridge (uDSB) to model cellular processes by reconstructing cellular trajectories, reveal gene expression dynamics, and recover cell population changes. The VAE maps input time-series single-cell data to a continuous latent space, where trajectories are reconstructed by solving the Schrödinger bridge problem using forward-backward non-linear stochastic differential equations (SDEs). A drift function in the SDEs captures deterministic gene expression trends. An additional neural network estimates time-varying kill rates of single cells along trajectories, enabling recovery of cell population changes.

![alt text](https://github.com/sayali7/ARTEMIS/blob/main/paper_figures/Figure1.png?raw=true)

## Requirements
Our code has been tested in Python 3.8 & 3.10 on Linux Ubuntu machines. Required packages are:
- JAX
- Haiku
- Optax
- Pandas
- Numpy
- Scipy
- OTT

## Usage
Tutorial notebooks to train model and downstream analyses are in [ARTEMIS/notebooks]{https://github.com/sayali7/ARTEMIS/tree/main/notebooks}.

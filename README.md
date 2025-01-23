# ARTEMIS

## Abstract
Cellular processes like development, differentiation, and disease progression are highly complex and dynamic (e.g., gene expression). These processes often undergo cell population changes driven by cell birth, proliferation, and death. Single-cell sequencing enables gene expression measurement at single-cell resolution, allowing us to decipher cellular and molecular dynamics underlying these processes. However, the high costs and destructive nature of sequencing restrict observations to snapshots of unaligned cells at discrete timepoints, limiting our understanding of these processes and complicating the reconstruction of cellular trajectories.
To address this challenge, we propose ARTEMIS, a generative model integrating a variational autoencoder (VAE) with unbalanced diffusion schrödinger bridge (uDSB) to model cellular processes by reconstructing cellular trajectories, reveal gene expression dynamics, and recover cell population changes. The VAE maps input time-series single-cell data to a continuous latent space, where trajectories are reconstructed by solving the Schrödinger bridge problem using forward-backward non-linear stochastic differential equations (SDEs). A drift function in the SDEs captures deterministic gene expression trends. An additional neural network estimates time-varying kill rates of single cells along trajectories, enabling recovery of cell population changes.

![alt text](https://github.com/sayali7/ARTEMIS/blob/main/paper_figures/Figure1.png?raw=true)

## Requirements
Our code has been tested in Python 3.8 & 3.10 on Linux Ubuntu, both on machines with CPU and with GPU NVIDIA RTX A6000 (recommended). Required packages are:
- JAX
- Haiku
- Optax
- Pandas
- Numpy
- Scipy
- OTT

## Data
1. The pancreatic data, can be downloaded from GEO (\href{https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114412}{GSE114412})\cite{veres2019charting}.
2. The raw zebrafish data can be downloaded from \url{https://figshare.com/articles/dataset/Raw_and_processed_data_of_three_scRNA-seq_datasets_/25601610/1?file=45647244}\cite{zhang2024scnode}. It can also be downloaded from \href{https://singlecell.broadinstitute.org/single_cell/study/SCP162/single-cell-reconstruction-of-developmental-trajectories-during-zebrafish-embryogenesis}{Broad Single Cell Portal} with identifier SCP126 \cite{farrell2018single}.
3. The TGFB1-induced EMT from A549 lung cancer cell data can be downloaded from \url{https://github.com/dpcook/emt_dynamics} \cite{cook2020context}.

## Usage
Tutorial notebooks to train model and downstream analyses are in [ARTEMIS/notebooks](https://github.com/sayali7/ARTEMIS/tree/main/notebooks).

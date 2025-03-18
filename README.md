# ARTEMIS

## Abstract
Cellular processes like development, differentiation, and disease progression are highly complex and dynamic (e.g., gene expression). These processes often undergo cell population changes driven by cell birth, proliferation, and death. Single-cell sequencing enables gene expression measurement at single-cell resolution, allowing us to decipher cellular and molecular dynamics underlying these processes. However, the high costs and destructive nature of sequencing restrict observations to snapshots of unaligned cells at discrete timepoints, limiting our understanding of these processes and complicating the reconstruction of cellular trajectories.
To address this challenge, we propose ARTEMIS, a generative model integrating a variational autoencoder (VAE) with unbalanced diffusion schrödinger bridge (uDSB) to model cellular processes by reconstructing cellular trajectories, reveal gene expression dynamics, and recover cell population changes. The VAE maps input time-series single-cell data to a continuous latent space, where trajectories are reconstructed by solving the Schrödinger bridge problem using forward-backward non-linear stochastic differential equations (SDEs). A drift function in the SDEs captures deterministic gene expression trends. An additional neural network estimates time-varying kill rates of single cells along trajectories, enabling recovery of cell population changes.

![alt text](https://github.com/sayali7/ARTEMIS/blob/main/paper_figures/Figure1.png?raw=true)

## Requirements
Our code has been tested in Python 3.8 & 3.10 on Linux Ubuntu (20.04,22.04), both on machines with CPU and with GPU NVIDIA RTX A6000. Since the code is implemented on JAX, we recommend users to train on GPU to take advantage of the jax jit-compilation. The average runtime on the gpu is ~15 minutes, whie that on the cpu is ~30 minutes. Main packages required for training are:
- JAX
- Flax
- Haiku
- Optax
- Pandas
- Numpy
- Scipy
- OTT

All packages with versions, including those used in preprocessing and analysis, can be downloaded using: [requirements.txt](https://github.com/sayali7/ARTEMIS/blob/main/requirements.txt).
## Data
1. The pancreatic data, can be downloaded from GEO [GSE114412](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114412) [1].
2. The raw zebrafish data can be downloaded from [https://figshare.com/articles/dataset/Raw_and_processed_data_of_three_scRNA-seq_datasets_/25601610/1?file=45647244](https://figshare.com/articles/dataset/Raw_and_processed_data_of_three_scRNA-seq_datasets_/25601610/1?file=45647244) [4]. It can also be downloaded from [Broad Single Cell Portal](https://singlecell.broadinstitute.org/single_cell/study/SCP162/single-cell-reconstruction-of-developmental-trajectories-during-zebrafish-embryogenesis) with identifier SCP126 [2].
3. The TGFB1-induced EMT from A549 lung cancer cell data can be downloaded from [https://github.com/dpcook/emt_dynamics](https://github.com/dpcook/emt_dynamics) [3].

## Usage
Tutorial notebooks to train model and downstream analyses are in [ARTEMIS/notebooks](https://github.com/sayali7/ARTEMIS/tree/main/notebooks).

## References
<a id="1">[1]</a>
Adrian Veres, Aubrey L Faust, Henry L Bushnell, Elise N
Engquist, Jennifer Hyoje-Ryu Kenty, George Harb, Yeh-
Chuin Poh, Elad Sintov, Mads G¨urtler, Felicia W Pagliuca,
et al. Charting cellular identity during human in vitro β-cell
differentiation. Nature, 569(7756):368–373, 2019.

<a id="2">[2]</a>
Jeffrey A Farrell, Yiqun Wang, Samantha J Riesenfeld,
Karthik Shekhar, Aviv Regev, and Alexander F
Schier. Single-cell reconstruction of developmental
trajectories during zebrafish embryogenesis. Science,
360(6392):eaar3131, 2018.

<a id="3">[3]</a>
David P Cook and Barbara C Vanderhyden. Context
specificity of the emt transcriptional response. Nature
communications, 11(1):2142, 2020.

<a id="4">[4]</a>
Jiaqi Zhang, Erica Larschan, Jeremy Bigness, and
Ritambhara Singh. scnode: generative model for temporal
single cell transcriptomic data prediction. Bioinformatics,
40(Supplement 2):ii146–ii154, 2024

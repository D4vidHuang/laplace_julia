# Laplace Redux Reproduction

This repository contains code to reproduce Table 1 from the paper:

*"Laplace Redux - Effortless Bayesian Deep Learning"* by Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, and Philipp Hennig.

## Overview

The script reproduces the out-of-distribution detection performance comparison of different Bayesian and non-Bayesian methods on MNIST dataset:
- Maximum A Posteriori (MAP)
- Deep Ensembles (DE)
- Variational Bayes (VB)
- Hamiltonian Monte Carlo (HMC)
- Stochastic Weight Averaging-Gaussian (SWAG)
- Laplace Approximation (LA)
- Improved Laplace Approximation (LA*)

Performance is measured using two metrics:
- Confidence (lower is better): average maximum prediction probability
- AUROC (higher is better): area under ROC curve for OOD detection

## Requirements

To run this code, you'll need Julia 1.6+ and the following packages:
- Flux
- Statistics
- Random
- Optimisers
- LaplaceRedux
- Plots
- MLDatasets
- BSON
- LinearAlgebra
- ROCAnalysis
- CUDA (optional, for GPU acceleration)
- ProgressMeter
- DataFrames
- CSV
- Printf
- TaijaPlotting (for calibration plots)

You can install the required packages using:

```julia
using Pkg
Pkg.add(["Flux", "Optimisers", "LaplaceRedux", "Plots", "MLDatasets", "BSON", 
         "LinearAlgebra", "ROCAnalysis", "CUDA", "ProgressMeter", "DataFrames", 
         "CSV", "Printf", "TaijaPlotting"])
```

## Running the Code

To reproduce Table 1 from the paper:

```bash
julia src/reproduce_table1.jl
```

This will:
1. Load MNIST and out-of-distribution datasets
2. Train and evaluate all 7 methods
3. Print the results formatted as a table
4. Save the results to `table1_results.csv`

## Notes

- The script uses a simplified version of MNIST for faster execution
- For HMC, we use a simplified implementation that approximates the full HMC procedure
- The different methods are run for a small number of iterations to speed up execution; increase epochs for better results

## Citation

If you find this code useful, please cite the original paper:

```
@article{daxberger2021laplace,
  title={Laplace Redux--Effortless Bayesian Deep Learning},
  author={Daxberger, Erik and Kristiadi, Agustinus and Immer, Alexander and Eschenhagen, Runa and Bauer, Matthias and Hennig, Philipp},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={3454--3465},
  year={2021}
}
```

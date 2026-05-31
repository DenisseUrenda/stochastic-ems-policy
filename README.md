# Stochastic Energy Management System
Energy Management System with grid and battery control under stochastic electricity prices and EV arrival/departure dynamics.

## Overview

This repository contains a simplified Energy Management System (EMS) for EV charging under stochastic electricity prices and stochastic EV arrival/departure processes.

The current implementation considers:

- grid interaction
- battery dynamics
- homogeneous EVs
- stochastic electricity prices
- stochastic EV arrival/departure behavior
- neural-network-based control policies

The repository is intended as supplementary material for the associated research work and contains:

- EMS implementation details
- policy training procedures
- experimental studies
- trajectory analysis
- additional figures and results

### EMS Architecture

The following diagram illustrates the general EMS structure considered in this repository, including stochastic inputs, grid interaction, battery dynamics, and EV charging allocation.

<p align="center">
  <img src="figures/system_architecture.png" width="450">
</p>

### Optimization Workflow

The following diagram summarizes the sequential optimization and rollout structure used during policy training and trajectory generation.
<p align="center">
  <img src="figures/nn-diagram.png" width="450">
</p>

## Repository Structure

```bash
.
├── data/
├── figures/
├── notebooks/ 
│   ├── 00_sanity_checks.ipynb
│   ├── 01_training_demo.ipynb
│   ├── 02_policy_comparison.ipynb
│   └── 03_lambda_experiments.ipynb
├── source/
├── styles/
├── README.md
├── requirements.txt
└── .gitignore
```

The `source/` directory contains the core implementation of the Energy Management System, including:

- diagnostics
- plotting utilities
- neural policies
- simulation components
- stochastic processes
- system dynamics
- training routines
- general utilities

## Data

The experiments use a preprocessed electricity price time series derived from the ERCOT Real-Time Settlement Point Prices (SPP) database.

The file expected by the notebooks is:

```text
data/eprice_test.csv
```

This file is not distributed with the repository.

Expected file format:

- CSV file
- Single column
- No header
- Hourly electricity prices
- Original units: $/MWh
- Internally converted to $/kWh
- At least 48 observations

The first 24 observations (one day) are used as historical data to initialize the stochastic price model, while the following 24 observations are used as the future reference trajectory for a complete 24-hour rollout:

```python
energy_prices = read_csv("../data/eprice_test.csv")
price_history = energy_prices[:params.T]
future_price = energy_prices[params.T:]
```

The stochastic price process is modeled using a SARIMA model with the following orders:

```python
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 24)
```

Although the EMS simulation can use a smaller internal time step (e.g., `dt = 0.5` hours), electricity prices are provided at hourly resolution. The implementation performs the required interpolation internally.


## Experimental Studies

### Experiment 1 — Policy Comparison

This experiment compares three control strategies under identical stochastic scenarios:

- Heuristic policy
- Untrained neural policy
- Trained neural policy

The objective is to evaluate whether training produces meaningful control behavior and cost improvements relative to simple baselines.

Generated outputs include:

- state trajectories
- control trajectories
- energy flow trajectories
- interval costs
- cumulative cost comparisons
- training curves

Notebook:

```bash
notebooks/02_policy_comparison.ipynb
```

### Experiment 2 — Lambda Experiments

This experiment studies the effect of introducing a risk-sensitive term into the objective function.

The parameter λ penalizes positive interval costs during optimization, encouraging more conservative decision-making.

Different λ values are trained and evaluated under identical stochastic scenarios.

Generated outputs include:

- policy checkpoints
- training convergence curves
- state trajectory comparisons
- control trajectory comparisons
- energy flow comparisons
- cost analysis across λ values

Notebook:

```bash
notebooks/03_lambda_experiments.ipynb
```

## Checkpoints

Training checkpoints are generated locally during experimentation and are not included in the repository.

Typical checkpoint files include:

- trained_policy.pt
- untrained_policy.pt
- policy_lambda_0_0.pt
- policy_lambda_0_1.pt
- policy_lambda_0_25.pt
- policy_lambda_0_5.pt

Users running the notebooks will generate these files automatically in their local environment.

## Data Availability

The electricity price dataset used in the experiments is derived from the ERCOT Real-Time Settlement Point Prices (SPP) database.

The preprocessed file used during development is not distributed with this repository. Users interested in reproducing the experiments should obtain the original data directly from ERCOT and generate the corresponding input time series described in the Data section.

ERCOT Real-Time Settlement Point Prices (SPP):
https://www.ercot.com/content/cdr/html/real_time_spp.html


## Notes

- The repository follows a research-oriented workflow built around reusable Python modules and Jupyter notebooks.
- Figures are generated directly within each experiment notebook.
- Checkpoints are saved automatically to support reproducible evaluations.
- The code is intended as supplementary material accompanying the associated research work.


## Author
**Denisse Urenda Castañeda**

PhD Student in Data Science — UTEP

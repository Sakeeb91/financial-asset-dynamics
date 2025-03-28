# Financial Asset Dynamics Modeling

A Python project for exploring, implementing, and applying Stochastic Differential Equations (SDEs) for modeling financial asset dynamics.

## Features

- Implementation of Geometric Brownian Motion (GBM)
- Implementation of Ornstein-Uhlenbeck (OU) process
- Advanced simulation methods (Euler-Maruyama, Milstein)
- Parameter calibration from historical data
- Correlated asset simulation
- Jump diffusion modeling (optional)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd sde_asset_modeling
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Project Structure

- `data/`: Directory for storing raw or processed data
- `notebooks/`: Jupyter notebooks for exploration and presentation
- `src/sde_asset_modeling/`: Main source code
  - `models/`: SDE model definitions
  - `simulation/`: Simulation schemes
  - `calibration/`: Parameter estimation methods
  - `utils/`: Utility functions
- `tests/`: Unit tests

## Usage

The project is organized as a series of Jupyter notebooks that demonstrate each aspect of SDE modeling:

1. `01_GBM_Simulation.ipynb`: Basic simulation of GBM
2. `02_OU_Simulation.ipynb`: Basic simulation of OU process
3. `03_Milstein_Comparison.ipynb`: Comparison of simulation methods
4. `04_Calibration.ipynb`: Parameter estimation from real data
5. `05_Correlated_Simulation.ipynb`: Simulation of correlated assets
6. `06_Jump_Diffusion.ipynb`: Optional extension for jump diffusion modeling

## License

[MIT License](LICENSE)

# SDE Asset Modeling - Directory Structure

This project is organized into the following structure to keep simulations, dashboards, and plots separate and well-organized.

## Directory Structure

```
sde_asset_modeling/
├── simulations/           # Simulation scripts for each model
│   ├── gbm_simulation.py  # Geometric Brownian Motion
│   ├── ou_simulation.py   # Ornstein-Uhlenbeck process
│   ├── jump_diffusion_simulation.py
│   ├── heston_simulation.py
│   ├── sabr_simulation.py
│   └── correlated_simulation.py
│
├── dashboards/            # Dashboard creation scripts
│   ├── gbm_dashboard.py
│   ├── ou_dashboard.py
│   ├── jump_diffusion_dashboard.py
│   ├── heston_dashboard.py
│   ├── sabr_dashboard.py
│   └── correlated_dashboard.py
│
├── plots/                 # Generated plots organized by model
│   ├── gbm/               # GBM simulation plots
│   ├── ou/                # OU process plots
│   ├── jump_diffusion/    # Jump Diffusion model plots
│   ├── heston/            # Heston model plots
│   ├── sabr/              # SABR model plots
│   └── correlated/        # Correlated assets plots
│
├── src/                   # Core implementation code
│   └── sde_asset_modeling/
│       ├── models/        # SDE model implementations
│       ├── simulation/    # Simulation engines and numerical methods
│       └── utils/         # Utility functions
│
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
└── data/                  # Data files
```

## Running Simulations

To run a simulation for a specific model, use the corresponding script in the `simulations/` directory:

```
python simulations/gbm_simulation.py  # Run GBM simulation
```

## Creating Dashboards

To create a dashboard for a specific model, use the corresponding script in the `dashboards/` directory:

```
python dashboards/gbm_dashboard.py  # Create GBM dashboard
```

Each dashboard script will check if the required plots exist and run the simulation if needed.

## Model Overview

1. **Geometric Brownian Motion (GBM)**
   - Standard model for stock price movements
   - Constant drift and volatility

2. **Ornstein-Uhlenbeck (OU) Process**
   - Mean-reverting stochastic process
   - Used for interest rates, volatility, and spread modeling

3. **Jump Diffusion (Merton) Model**
   - GBM with added discrete jumps
   - Captures market shocks and sudden price changes

4. **Heston Stochastic Volatility Model**
   - Extends GBM with stochastic volatility
   - Captures volatility clustering and smile effects

5. **SABR Model**
   - Stochastic Alpha Beta Rho model
   - Used for interest rate derivatives and volatility smile

6. **Correlated Assets Model**
   - Multi-asset simulation with correlation structure
   - Used for portfolio analysis and diversification studies 
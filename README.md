# Financial Asset Dynamics

A comprehensive Python library for modeling and simulating stochastic processes commonly used in financial mathematics, quantitative finance, and derivative pricing.

## Overview

This project implements various stochastic differential equation (SDE) models used to simulate financial asset price dynamics, volatility surfaces, and other financial metrics. It includes dashboards for interactive visualization and analysis of simulated paths.

## Models Implemented

- **Geometric Brownian Motion (GBM)**: The standard model for stock price dynamics
- **Ornstein-Uhlenbeck (OU) Process**: Mean-reverting stochastic process useful for interest rates
- **Heston Model**: Stochastic volatility model for capturing volatility clustering and skew
- **SABR Model**: Stochastic Alpha-Beta-Rho model for modeling volatility smiles in interest rate derivatives
- **Jump-Diffusion Model**: Extensions of GBM with discontinuous jumps for modeling market shocks
- **Correlated Assets**: Simulation of assets with correlation structures

## Features

- Efficient Monte Carlo simulations for all implemented models
- Parameter calibration functionality for some models
- Interactive dashboards for visualization using Python plotting libraries
- Model-specific analytics and option pricing capability
- Support for various financial metrics and statistics

## Installation

```bash
# Clone the repository
git clone https://github.com/Sakeeb91/financial-asset-dynamics.git
cd financial-asset-dynamics

# Install dependencies
pip install -r sde_asset_modeling/requirements.txt

# Install the package in development mode
cd sde_asset_modeling
pip install -e .
```

## Usage Examples

### Running a GBM Simulation

```python
from sde_asset_modeling.models.gbm import GBMModel
from sde_asset_modeling.simulation.engine import simulate_paths

# Create a GBM model instance
mu = 0.05  # Drift
sigma = 0.2  # Volatility
x0 = 100  # Initial price
model = GBMModel(mu, sigma, x0)

# Simulation parameters
t_span = (0.0, 1.0)  # Time span (1 year)
dt = 1/252  # Daily steps
n_paths = 1000  # Number of paths

# Run simulation
t_points, paths = simulate_paths(model, t_span, dt, n_paths)
```

### Running a SABR Simulation

```python
from sde_asset_modeling.models.sabr import SABRModel

# Create a SABR model
alpha = 0.15  # Initial volatility
beta = 0.5  # CEV parameter
nu = 0.4  # Volatility of volatility
rho = -0.3  # Correlation
f0 = 0.03  # Initial forward rate
alpha0 = alpha  # Initial volatility

# Create the model
model = SABRModel(alpha, beta, nu, rho, f0, alpha0)

# Run the simulation script
python sde_asset_modeling/sabr_simulation.py
```

### Creating Dashboards

```bash
# Run the dashboard for GBM
python sde_asset_modeling/create_dashboard.py

# Run the dashboard for Heston model
python sde_asset_modeling/create_heston_dashboard.py

# Run the dashboard for OU process
python sde_asset_modeling/create_ou_dashboard.py
```

## Project Structure

```
sde_asset_modeling/
├── src/sde_asset_modeling/       # Core package
│   ├── models/                   # Model implementations
│   ├── simulation/               # Simulation engines
│   ├── calibration/              # Parameter calibration
│   └── utils/                    # Utility functions
├── dashboards/                   # Interactive visualizations
├── notebooks/                    # Example Jupyter notebooks
├── docs/                         # Documentation and results
├── plots/                        # Generated plots
├── tests/                        # Unit tests
└── requirements.txt              # Dependencies
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

This project uses mathematical models and techniques from quantitative finance literature, with implementations inspired by various open-source libraries and academic papers. 
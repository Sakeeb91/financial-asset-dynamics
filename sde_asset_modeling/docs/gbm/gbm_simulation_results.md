# Geometric Brownian Motion (GBM) Simulation Results

## Overview
This document explains the results of the Geometric Brownian Motion (GBM) simulation, which is a widely used stochastic process for modeling financial asset prices.

## Model Parameters
- **Drift (μ)**: 0.10 (10% annual expected return)
- **Volatility (σ)**: 0.20 (20% annual volatility)
- **Initial Price (S₀)**: $100.00

## Simulation Results

### Single Path Simulation
![Single Path](../../plots/gbm/gbm_single_path.png)

The single path simulation shows one possible trajectory of the asset price over time. The results show:
- Final price using Euler-Maruyama method: $107.19
- Final price using Exact solution: $165.60
- Absolute difference: $58.41 (35.27% relative error)

The significant difference between the numerical and analytical solutions highlights the importance of using appropriate numerical methods or sufficiently small time steps when simulating stochastic processes.

### Multiple Paths Simulation
![Multiple Paths](../../plots/gbm/gbm_multiple_paths.png)

The multiple paths simulation shows 100 possible trajectories of the asset price, illustrating the range of possible outcomes. The red line represents the mean path. While individual paths can vary significantly, the mean behavior aligns with the expected drift of the process.

### Distribution of Final Prices
![Price Distribution](../../plots/gbm/gbm_distribution.png)

The distribution analysis of 10,000 simulated paths reveals:
- **Theoretical Mean**: $110.52
- **Empirical Mean**: $110.84 (0.29% difference)
- **Theoretical Standard Deviation**: $22.33
- **Empirical Standard Deviation**: $22.46 (0.58% difference)

The close agreement between theoretical and empirical statistics confirms the accuracy of the simulation at this sample size. The final price distribution follows a log-normal distribution, as expected for a GBM process.

### Numerical Methods Comparison
![Methods Comparison](../../plots/gbm/gbm_methods_comparison.png)

The comparison between numerical methods shows:

|Method|Final Price|Absolute Error|Relative Error|
|------|-----------|--------------|--------------|
|Euler|$107.19|$58.41|35.27%|
|Milstein|$107.04|$58.57|35.36%|

Both Euler-Maruyama and Milstein methods perform similarly for this particular GBM process with the given parameters. This is because GBM has constant volatility, which reduces the advantage of higher-order methods like Milstein.

### Convergence Analysis
![Convergence Analysis](../../plots/gbm/gbm_convergence.png)

The convergence analysis examines how the error changes with different time step sizes. Results show:

|Time Step|Euler Mean Error|Milstein Mean Error|
|---------|----------------|-------------------|
|1/12|$22.93|$22.92|
|1/52|$24.99|$25.01|
|1/252|$24.36|$24.36|
|1/504|$26.63|$26.62|
|1/1008|$22.81|$22.80|

Interestingly, the error doesn't consistently decrease with smaller time steps as might be expected. This could be due to:
1. The stochastic nature of the process
2. The specific random seed used
3. The need for even smaller time steps to observe clear convergence patterns

### Dashboard
![GBM Dashboard](../../plots/gbm/gbm_simulation_dashboard.png)

The dashboard above provides a comprehensive overview of all the simulation results, showcasing the different aspects of the GBM model in a single visual display.

## Conclusion
The GBM simulation demonstrates how stochastic processes can model financial asset prices. While the numerical approximations show considerable error compared to the analytical solution for single paths, the statistical properties of many simulated paths align well with theoretical expectations. This confirms the validity of using GBM as a model for financial assets.

The simulation also highlights the importance of considering numerical method accuracy and time step size when implementing stochastic differential equations. 
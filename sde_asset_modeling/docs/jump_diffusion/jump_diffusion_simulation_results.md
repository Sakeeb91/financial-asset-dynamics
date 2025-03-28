# Jump Diffusion (Merton) Simulation Results

## Overview
This document explains the results of the Merton Jump Diffusion model simulation, which extends the standard Geometric Brownian Motion (GBM) by incorporating sudden jumps in asset prices. The jump diffusion model is particularly useful for capturing market behavior during events like earnings announcements, economic crises, or other significant news that cause abrupt price movements.

## Model Parameters
- **Drift (μ)**: 0.10 (10% annual expected return)
- **Volatility (σ)**: 0.20 (20% annual volatility)
- **Jump Intensity (λ)**: 5.0 (expected 5 jumps per year)
- **Mean Jump Size**: -4.88% (logarithmic mean of -0.05)
- **Jump Size Standard Deviation**: 0.10
- **Initial Price (S₀)**: $100.00

## Simulation Results

### Single Path Simulation
![Single Path](../../plots/jump_diffusion/jump_diffusion_single_path.png)

The single path simulation compares a standard GBM path (blue line) with a jump diffusion path (red line). Key observations:
- The jump diffusion path exhibits sudden discontinuous movements (jumps) that are not present in the standard GBM path
- These jumps are marked with green vertical lines
- Jump sizes vary according to the log-normal distribution specified by the mean and standard deviation parameters
- While most jumps in this simulation are negative (reflecting the negative mean jump size parameter), positive jumps can also occur

### Multiple Paths Simulation
![Multiple Paths](../../plots/jump_diffusion/jump_diffusion_multiple_paths.png)

The multiple paths simulation shows 50 possible trajectories for the asset price under the jump diffusion model:
- Each blue line represents a different possible price path
- The red line shows the mean path across all simulations
- The dotted horizontal line indicates the initial price ($100)
- The spread of outcomes is wider than in a standard GBM model due to the additional uncertainty introduced by jumps
- Some paths exhibit significant drops due to the negative jump bias in our parameters

### Distribution of Final Prices
![Price Distribution](../../plots/jump_diffusion/jump_diffusion_distribution.png)

The distribution analysis of 5,000 simulated paths reveals several important features:
- **Mean**: The mean final price reflects both the drift component and the impact of jumps
- **Standard Deviation**: The standard deviation is larger than would be expected from a pure GBM process with the same drift and volatility
- **Skewness**: The distribution is negatively skewed due to the negative mean jump size
- **Excess Kurtosis**: The distribution exhibits fat tails (higher kurtosis than normal distribution), which is characteristic of jump processes
- The comparison with the normal distribution (red line) illustrates how the jump component creates a more leptokurtic (peaked) distribution with fatter tails

### Comparison of Jump Parameters
![Jump Parameters Comparison](../../plots/jump_diffusion/jump_diffusion_comparison.png)

This analysis compares different jump parameter configurations:
- **Rare Jumps (λ=1)**: Fewer but significant jumps
- **Frequent Small Jumps (λ=10)**: Many smaller jumps, creating a more continuous but volatile path
- **Medium Frequency Large Jumps (λ=3)**: Less frequent but larger jumps, leading to more dramatic price changes
- **Positive Jumps (λ=5)**: Same frequency as the base case but with positive mean jump size, leading to upward bias

Each parameter set creates unique price dynamics, demonstrating the model's flexibility in capturing different market conditions and asset behaviors.

### Option Pricing Analysis
![Option Prices](../../plots/jump_diffusion/jump_diffusion_option_prices.png)

The option pricing analysis compares Black-Scholes and Jump Diffusion models:
- **Left Panel**: Call option prices across different strike prices
- **Right Panel**: Put option prices across different strike prices
- The jump diffusion model consistently prices options higher than the Black-Scholes model, especially for out-of-the-money options
- This reflects the additional risk premium for rare but significant events (jumps) that the standard Black-Scholes model doesn't account for
- The difference is particularly prominent for out-of-the-money put options, which are more affected by the negative jump risk

### Dashboard Summary
![Jump Diffusion Dashboard](../../plots/jump_diffusion/jump_diffusion_dashboard.png)

The dashboard provides a comprehensive overview of all simulation results, showcasing the different aspects of the Merton Jump Diffusion model in a single visual display.

## Conclusion
The Merton Jump Diffusion model extends the standard GBM by introducing sudden jumps in asset prices, making it more realistic for modeling financial markets that exhibit occasional rapid movements. The simulation demonstrates several important features:

1. **Fat Tails**: The jump component creates fatter tails in return distributions, better matching empirical market data
2. **Skewness**: The model can capture asymmetric return distributions through the jump size parameters
3. **Volatility Smiles**: The option pricing analysis shows how jump risk affects option prices differently across strike prices, helping explain the volatility smile observed in options markets
4. **Risk Management**: The model highlights the importance of accounting for jump risk in portfolio management and risk assessment

While more complex than standard GBM, the jump diffusion model provides a more accurate representation of financial markets, particularly for assets that exhibit occasional discontinuous price movements. 
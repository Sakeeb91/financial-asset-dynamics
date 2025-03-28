# Ornstein-Uhlenbeck Process Simulation Results

## Model Overview

The Ornstein-Uhlenbeck (OU) process is a stochastic process that describes the velocity of a massive Brownian particle under the influence of friction. In finance, it is widely used to model mean-reverting processes such as interest rates, volatility, and spread trading.

The OU process is described by the following stochastic differential equation (SDE):

$dX_t = \theta(\mu - X_t)dt + \sigma dW_t$

Where:
- $X_t$ is the value of the process at time $t$
- $\theta$ is the speed of mean reversion
- $\mu$ is the long-term mean
- $\sigma$ is the volatility
- $dW_t$ is the increment of a Wiener process

## Simulation Parameters

For this simulation, we used the following parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\theta$ | 5.0 | Speed of mean reversion |
| $\mu$ | 0.07 | Long-term mean (7%) |
| $\sigma$ | 0.02 | Volatility |
| $X_0$ | 0.10 | Initial value (10%) |

The half-life of deviations in this model is $\frac{\ln(2)}{\theta} = 0.1386$ years, which means it takes about 0.1386 years for a deviation from the mean to reduce by half.

## Simulation Results

### 1. Single Path Analysis

![Single Path](../plots/ou_single_path.png)

This plot shows a single realization of the OU process over a one-year period, comparing the Euler-Maruyama numerical solution with the exact analytical solution.

Key observations:
- The process starts at 10% and reverts toward the long-term mean of 7%
- The numerical solution (Euler-Maruyama) closely matches the exact solution
- The process exhibits random fluctuations around the mean, constrained by the mean-reverting force

### 2. Multiple Paths Analysis

![Multiple Paths](../plots/ou_multiple_paths.png)

This visualization shows 100 different simulated paths of the OU process. The red line represents the mean path.

Key observations:
- All paths tend to revert to the long-term mean of 7%
- The mean path shows a smooth descent from the initial value to the long-term mean
- The dispersion of paths illustrates the stochastic nature of the process

### 3. Distribution Analysis

![Distribution Analysis](../plots/ou_distributions.png)

This figure shows the distribution of values at different time points (t=0.1, t=0.5, t=1.0) and the stationary distribution.

Key observations:
- Initially, the distribution is tightly centered around the starting point
- As time progresses, the distribution shifts toward the long-term mean and spreads out
- At each time point, the distribution is approximately normal
- The stationary distribution (as t→∞) is normal with mean μ and variance σ²/(2θ)

### 4. Numerical Methods Comparison

![Methods Comparison](../plots/ou_methods_comparison.png)

This chart compares different numerical methods for solving the OU SDE:

1. **Euler-Maruyama method**: A simple first-order method
2. **Milstein method**: A higher-order method that includes additional terms
3. **Exact solution**: The analytical solution for the OU process

Key observations:
- For the OU process with constant volatility, both numerical methods perform similarly
- The difference from the exact solution is very small (less than 0.06%)
- The Milstein method offers no significant advantage over Euler-Maruyama for this particular SDE

### 5. Mean Reversion Demonstration

![Mean Reversion](../plots/ou_mean_reversion.png)

This plot demonstrates the mean-reverting property of the OU process by simulating paths with different starting points.

Key observations:
- Paths starting below the long-term mean tend to rise toward it
- Paths starting above the long-term mean tend to fall toward it
- All paths converge to fluctuate around the long-term mean
- The speed of reversion is governed by the parameter θ

### 6. Convergence Analysis

![Convergence Analysis](../plots/ou_convergence.png)

This analysis shows how the numerical error changes with different time step sizes.

Key observations:
- Both methods show approximately first-order convergence (error ∝ dt)
- The error decreases as the time step size decreases
- For practical applications, a time step of 1/252 (daily) provides good accuracy

## Applications in Finance

The OU process is particularly useful in finance for modeling:

1. **Interest rates**: The Vasicek model for interest rates is an OU process
2. **Volatility**: Mean-reverting stochastic volatility models
3. **Spread trading**: Modeling the spread between related securities
4. **Commodity prices**: Some commodities exhibit mean-reverting behavior

## Conclusions

The OU process provides a powerful tool for modeling mean-reverting behavior in financial markets. Key properties demonstrated in this simulation include:

1. **Mean Reversion**: The process gravitates toward a long-term mean
2. **Normal Distributions**: Value distributions are approximately normal
3. **Stationary Distribution**: As time increases, the process approaches a stationary normal distribution
4. **Analytical Solution**: The OU process has an analytical solution, making it useful for testing numerical methods

These properties make the OU process valuable for risk management, derivatives pricing, and analyzing the dynamics of financial variables that display mean-reverting tendencies.

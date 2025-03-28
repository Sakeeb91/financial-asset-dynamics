#!/usr/bin/env python
"""
Geometric Brownian Motion (GBM) Simulation Script
This script demonstrates GBM simulation for financial asset modeling.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

# Define plot directory
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots', 'gbm')
os.makedirs(PLOT_DIR, exist_ok=True)

# Import modules from the project
from sde_asset_modeling.models.gbm import GeometricBrownianMotion
from sde_asset_modeling.simulation.simulators import euler_maruyama, milstein
from sde_asset_modeling.simulation.engine import SimulationEngine
try:
    from sde_asset_modeling.utils.plotting import setup_plotting_style, plot_paths, plot_distribution
    # Set plotting style
    setup_plotting_style()
except ImportError:
    print("Plotting utilities not available, using default matplotlib style")

# Define parameters
mu = 0.10  # Annual drift (expected return): 10%
sigma = 0.20  # Annual volatility: 20%
s0 = 100.0  # Initial stock price: $100

# Create GBM model
gbm = GeometricBrownianMotion(mu, sigma)

print(f"Geometric Brownian Motion Model Parameters:")
print(f"Drift (μ): {mu:.2f} (expected annual return of {mu*100:.1f}%)")
print(f"Volatility (σ): {sigma:.2f} (annual volatility of {sigma*100:.1f}%)")
print(f"Initial price (S₀): ${s0:.2f}")

# Define simulation parameters
t_span = (0.0, 1.0)  # 1 year
dt = 1/252  # Daily timesteps (252 trading days per year)

# Create simulation engine
engine = SimulationEngine(gbm, s0, t_span, dt)

# 1. Simulate a single path using Euler-Maruyama
print("\n1. Simulating a single path...")
t, x_euler = engine.run_simulation(method='euler', n_paths=1, random_state=42)

# Simulate the same path using the analytical solution for comparison
t_exact, x_exact = engine.run_exact_solution(n_paths=1, random_state=42)

# Calculate final prices
print(f"Final price (Euler-Maruyama): ${x_euler[-1]:.2f}")
print(f"Final price (Exact solution): ${x_exact[-1]:.2f}")
print(f"Absolute difference: ${abs(x_euler[-1] - x_exact[-1]):.2f}")
print(f"Relative error: {abs(x_euler[-1] - x_exact[-1])/x_exact[-1]*100:.4f}%")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, x_euler, 'b-', label='Euler-Maruyama')
plt.plot(t_exact, x_exact, 'r--', label='Exact solution')
plt.xlabel('Time (years)')
plt.ylabel('Stock Price ($)')
plt.title('GBM Simulation: Single Path')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, 'gbm_single_path.png'))
print(f"Saved single path plot to '{os.path.join(PLOT_DIR, 'gbm_single_path.png')}'")

# 2. Simulate multiple paths
print("\n2. Simulating multiple paths...")
n_paths = 100
t, x_paths = engine.run_simulation(method='euler', n_paths=n_paths, random_state=42)

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(n_paths):
    plt.plot(t, x_paths[i], 'b-', alpha=0.1)
plt.plot(t, np.mean(x_paths, axis=0), 'r-', linewidth=2, label='Mean Path')
plt.axhline(s0, color='k', linestyle=':', label='Initial price')
plt.xlabel('Time (years)')
plt.ylabel('Stock Price ($)')
plt.title('GBM Simulation: Multiple Paths')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, 'gbm_multiple_paths.png'))
print(f"Saved multiple paths plot to '{os.path.join(PLOT_DIR, 'gbm_multiple_paths.png')}'")

# 3. Distribution of final prices
print("\n3. Analyzing distribution of final prices...")
n_paths = 10000
t, x_paths = engine.run_simulation(method='euler', n_paths=n_paths, random_state=42)

# Extract final prices
final_prices = x_paths[:, -1]

# Calculate theoretical parameters for log-normal distribution
T = t_span[1] - t_span[0]  # Total time period
theoretical_mean = s0 * np.exp(mu * T)
theoretical_std = s0 * np.exp(mu * T) * np.sqrt(np.exp(sigma**2 * T) - 1)

# Calculate empirical mean and standard deviation
empirical_mean = np.mean(final_prices)
empirical_std = np.std(final_prices)

# Print statistics
print(f"Statistics of final stock prices after {T} year(s):")
print(f"\nTheoretical:")
print(f"Mean: ${theoretical_mean:.2f}")
print(f"Standard deviation: ${theoretical_std:.2f}")
print(f"\nEmpirical (from {n_paths} simulations):")
print(f"Mean: ${empirical_mean:.2f}")
print(f"Standard deviation: ${empirical_std:.2f}")
print(f"\nRelative difference:")
print(f"Mean: {abs(empirical_mean - theoretical_mean)/theoretical_mean*100:.4f}%")
print(f"Standard deviation: {abs(empirical_std - theoretical_std)/theoretical_std*100:.4f}%")

# Plot histogram of final prices
plt.figure(figsize=(10, 6))
plt.hist(final_prices, bins=50, alpha=0.7, density=True, label='Simulated')
plt.axvline(empirical_mean, color='r', linestyle='-', label=f'Mean: ${empirical_mean:.2f}')
plt.axvline(s0, color='k', linestyle=':', label=f'Initial: ${s0:.2f}')
plt.xlabel('Stock Price ($)')
plt.ylabel('Probability Density')
plt.title('Distribution of Final Stock Prices')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, 'gbm_distribution.png'))
print(f"Saved distribution plot to '{os.path.join(PLOT_DIR, 'gbm_distribution.png')}'")

# 4. Compare numerical methods
print("\n4. Comparing numerical methods...")
results = engine.compare_methods(n_paths=1, random_state=42)

# Calculate error statistics compared to exact solution
exact_final = results['exact'][1][-1]
for method in ['euler', 'milstein']:
    method_final = results[method][1][-1]
    abs_error = abs(method_final - exact_final)
    rel_error = abs_error / exact_final * 100
    print(f"{method.capitalize()} method:")
    print(f"Final price: ${method_final:.2f}")
    print(f"Absolute error: ${abs_error:.6f}")
    print(f"Relative error: {rel_error:.6f}%\n")

# Plot comparison
plt.figure(figsize=(10, 6))
for method, (t, x) in results.items():
    plt.plot(t, x, label=method.capitalize())
plt.xlabel('Time (years)')
plt.ylabel('Stock Price ($)')
plt.title('Comparison of Numerical Methods for GBM')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, 'gbm_methods_comparison.png'))
print(f"Saved methods comparison plot to '{os.path.join(PLOT_DIR, 'gbm_methods_comparison.png')}'")

# 5. Convergence analysis
print("\n5. Performing convergence analysis...")
dt_values = np.array([1/12, 1/52, 1/252, 1/504, 1/1008])
errors = engine.compare_error(dt_values, final_time_only=True, n_paths=200, random_state=42)

# Print error statistics for each method and dt
print("Error statistics:")
print("\nEuler-Maruyama method:")
for i, dt in enumerate(dt_values):
    stats = errors['euler'][i]
    print(f"dt = {dt:.6f}: mean error = ${stats['mean']:.6f}, std = ${stats['std']:.6f}")

print("\nMilstein method:")
for i, dt in enumerate(dt_values):
    stats = errors['milstein'][i]
    print(f"dt = {dt:.6f}: mean error = ${stats['mean']:.6f}, std = ${stats['std']:.6f}")

# Plot convergence results
plt.figure(figsize=(10, 6))
for method, method_errors in errors.items():
    error_values = [e['mean'] for e in method_errors]
    plt.loglog(dt_values, error_values, 'o-', label=method.capitalize())

# Add reference lines for O(dt) and O(dt²) convergence
x_ref = np.logspace(np.log10(dt_values.min()), np.log10(dt_values.max()), 100)
y_ref1 = x_ref * (error_values[0] / dt_values[0])
y_ref2 = x_ref**2 * (error_values[0] / dt_values[0]**2)
plt.loglog(x_ref, y_ref1, 'k--', alpha=0.5, label='O(dt)')
plt.loglog(x_ref, y_ref2, 'k:', alpha=0.5, label='O(dt²)')

plt.xlabel('Time Step Size (dt)')
plt.ylabel('Mean Absolute Error')
plt.title('Convergence Analysis of Numerical Methods')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, 'gbm_convergence.png'))
print(f"Saved convergence analysis plot to '{os.path.join(PLOT_DIR, 'gbm_convergence.png')}'")

print(f"\nSimulation complete! All plots saved to '{PLOT_DIR}'.") 
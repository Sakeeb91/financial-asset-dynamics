#!/usr/bin/env python
"""
Ornstein-Uhlenbeck (OU) Process Simulation Script
This script demonstrates OU process simulation for financial asset modeling.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import stats

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules from the project
from sde_asset_modeling.models.ou import OrnsteinUhlenbeck
from sde_asset_modeling.simulation.simulators import euler_maruyama, milstein
from sde_asset_modeling.simulation.engine import SimulationEngine

try:
    from sde_asset_modeling.utils.plotting import setup_plotting_style, plot_paths, plot_distribution
    # Set plotting style
    setup_plotting_style()
except ImportError:
    print("Plotting utilities not available, using default matplotlib style")

# Define plots directory
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots', 'ou')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Define parameters
theta = 5.0   # Speed of mean reversion
mu = 0.07     # Long-term mean (e.g., long-term interest rate of 7%)
sigma = 0.02  # Volatility
x0 = 0.10     # Initial value (e.g., current interest rate of 10%)

# Create OU model
ou = OrnsteinUhlenbeck(theta, mu, sigma)

print(f"Ornstein-Uhlenbeck model parameters:")
print(f"Speed of mean reversion (θ): {theta:.2f}")
print(f"Long-term mean (μ): {mu:.2f} (7%)")
print(f"Volatility (σ): {sigma:.2f}")
print(f"Initial value (X₀): {x0:.2f} (10%)")
print(f"Half-life of deviations: {np.log(2)/theta:.4f} years")

# Define simulation parameters
t_span = (0.0, 1.0)  # 1 year
dt = 1/252  # Daily timesteps (252 trading days per year)

# Create simulation engine
engine = SimulationEngine(ou, x0, t_span, dt)

# 1. Simulate a single path using Euler-Maruyama
print("\n1. Simulating a single path...")
t, x_euler = engine.run_simulation(method='euler', n_paths=1, random_state=42)

# Simulate the same path using the analytical solution for comparison
t_exact, x_exact = engine.run_exact_solution(n_paths=1, random_state=42)

# Calculate final values
print(f"Final value (Euler-Maruyama): {x_euler[-1]:.4f}")
print(f"Final value (Exact solution): {x_exact[-1]:.4f}")
print(f"Absolute difference: {abs(x_euler[-1] - x_exact[-1]):.4f}")
print(f"Relative error: {abs(x_euler[-1] - x_exact[-1])/abs(x_exact[-1])*100:.4f}%")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, x_euler, 'b-', label='Euler-Maruyama')
plt.plot(t_exact, x_exact, 'r--', label='Exact solution')
plt.axhline(mu, color='g', linestyle='-', label=f'Long-term mean (μ={mu:.2f})')
plt.axhline(x0, color='k', linestyle=':', label=f'Initial value (X₀={x0:.2f})')
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.title('OU Process Simulation: Single Path')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOTS_DIR, 'ou_single_path.png'))
print(f"Saved single path plot to '{os.path.join(PLOTS_DIR, 'ou_single_path.png')}'")

# 2. Simulate multiple paths
print("\n2. Simulating multiple paths...")
n_paths = 100
t, x_paths = engine.run_simulation(method='euler', n_paths=n_paths, random_state=42)

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(n_paths):
    plt.plot(t, x_paths[i], 'b-', alpha=0.1)
plt.plot(t, np.mean(x_paths, axis=0), 'r-', linewidth=2, label='Mean Path')
plt.axhline(mu, color='g', linestyle='-', label=f'Long-term mean (μ={mu:.2f})')
plt.axhline(x0, color='k', linestyle=':', label=f'Initial value (X₀={x0:.2f})')
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.title('OU Process Simulation: Multiple Paths')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOTS_DIR, 'ou_multiple_paths.png'))
print(f"Saved multiple paths plot to '{os.path.join(PLOTS_DIR, 'ou_multiple_paths.png')}'")

# 3. Distribution of values at different time points
print("\n3. Analyzing distribution of values at different time points...")
n_paths = 10000
t, x_paths = engine.run_simulation(method='euler', n_paths=n_paths, random_state=42)

# Get stationary distribution parameters
stationary_mean, stationary_var = ou.stationary_distribution_parameters()
stationary_std = np.sqrt(stationary_var)

# Extract values at different time points
time_points = [0.1, 0.5, 1.0]  # 10%, 50%, 100% of the simulation period
time_indices = [int(tp/dt) for tp in time_points]

plt.figure(figsize=(12, 8))
for i, (tp, idx) in enumerate(zip(time_points, time_indices)):
    values = x_paths[:, idx]
    
    # Plot histogram
    plt.subplot(2, 2, i+1)
    plt.hist(values, bins=50, alpha=0.7, density=True, label=f't={tp}')
    
    # Overlay normal distribution
    x = np.linspace(values.min(), values.max(), 1000)
    mean_t = x0*np.exp(-theta*tp) + mu*(1-np.exp(-theta*tp))
    var_t = (sigma**2/(2*theta))*(1-np.exp(-2*theta*tp))
    std_t = np.sqrt(var_t)
    
    pdf = stats.norm.pdf(x, mean_t, std_t)
    plt.plot(x, pdf, 'r-', label=f'N({mean_t:.4f}, {std_t:.4f}²)')
    
    plt.axvline(mean_t, color='r', linestyle='--')
    plt.axvline(mu, color='g', linestyle='-', label=f'Long-term mean (μ={mu:.2f})')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.title(f'Distribution at t={tp}')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Add stationary distribution subplot
plt.subplot(2, 2, 4)
values = x_paths[:, -1]  # final values
plt.hist(values, bins=50, alpha=0.7, density=True, label='Final values')

# Overlay stationary distribution
x = np.linspace(values.min(), values.max(), 1000)
pdf = stats.norm.pdf(x, stationary_mean, stationary_std)
plt.plot(x, pdf, 'r-', label=f'N({stationary_mean:.4f}, {stationary_std:.4f}²)')
plt.axvline(stationary_mean, color='r', linestyle='--')
plt.axvline(mu, color='g', linestyle='-', label=f'Long-term mean (μ={mu:.2f})')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Stationary Distribution (t→∞)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'ou_distributions.png'))
print(f"Saved distribution plots to '{os.path.join(PLOTS_DIR, 'ou_distributions.png')}'")

# 4. Compare numerical methods
print("\n4. Comparing numerical methods...")
results = engine.compare_methods(n_paths=1, random_state=42)

# Calculate error statistics compared to exact solution
exact_final = results['exact'][1][-1]
for method in ['euler', 'milstein']:
    method_final = results[method][1][-1]
    abs_error = abs(method_final - exact_final)
    rel_error = abs_error / abs(exact_final) * 100
    print(f"{method.capitalize()} method:")
    print(f"Final value: {method_final:.4f}")
    print(f"Absolute error: {abs_error:.6f}")
    print(f"Relative error: {rel_error:.6f}%\n")

# Plot comparison
plt.figure(figsize=(10, 6))
for method, (t, x) in results.items():
    plt.plot(t, x, label=method.capitalize())
plt.axhline(mu, color='g', linestyle='-', label=f'Long-term mean (μ={mu:.2f})')
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.title('Comparison of Numerical Methods for OU Process')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOTS_DIR, 'ou_methods_comparison.png'))
print(f"Saved methods comparison plot to '{os.path.join(PLOTS_DIR, 'ou_methods_comparison.png')}'")

# 5. Mean-reversion demonstration
print("\n5. Demonstrating mean-reversion characteristic...")
# Create multiple simulations with different starting points
starting_points = [0.01, 0.04, 0.07, 0.10, 0.13, 0.16]
n_paths = 1

plt.figure(figsize=(10, 6))
for start in starting_points:
    # Create a new engine with the different starting point
    temp_engine = SimulationEngine(ou, start, t_span, dt)
    t, x = temp_engine.run_simulation(method='euler', n_paths=n_paths, random_state=42)
    plt.plot(t, x, label=f'X₀={start:.2f}')

plt.axhline(mu, color='g', linestyle='-', linewidth=2, label=f'Long-term mean (μ={mu:.2f})')
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.title('OU Process: Mean-Reversion Demonstration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOTS_DIR, 'ou_mean_reversion.png'))
print(f"Saved mean-reversion demonstration plot to '{os.path.join(PLOTS_DIR, 'ou_mean_reversion.png')}'")

# 6. Convergence analysis
print("\n6. Performing convergence analysis...")
dt_values = np.array([1/12, 1/52, 1/252, 1/504, 1/1008])
errors = engine.compare_error(dt_values, final_time_only=True, n_paths=200, random_state=42)

# Print error statistics for each method and dt
print("Error statistics:")
print("\nEuler-Maruyama method:")
for i, dt in enumerate(dt_values):
    stats = errors['euler'][i]
    print(f"dt = {dt:.6f}: mean error = {stats['mean']:.6f}, std = {stats['std']:.6f}")

print("\nMilstein method:")
for i, dt in enumerate(dt_values):
    stats = errors['milstein'][i]
    print(f"dt = {dt:.6f}: mean error = {stats['mean']:.6f}, std = {stats['std']:.6f}")

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
plt.savefig(os.path.join(PLOTS_DIR, 'ou_convergence.png'))
print(f"Saved convergence analysis plot to '{os.path.join(PLOTS_DIR, 'ou_convergence.png')}'")

print("\nSimulation complete! All plots saved to '{PLOTS_DIR}' directory.") 
#!/usr/bin/env python
"""
Jump Diffusion (Merton) Simulation Script
This script demonstrates simulation of asset prices using the Merton Jump Diffusion model.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import stats

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules from the project
from sde_asset_modeling.models.jump_diffusion import MertonJumpDiffusion
from sde_asset_modeling.simulation.simulators import euler_maruyama, generate_paths
from sde_asset_modeling.simulation.engine import SimulationEngine

try:
    from sde_asset_modeling.utils.plotting import setup_plotting_style
    # Set plotting style
    setup_plotting_style()
except ImportError:
    print("Plotting utilities not available, using default matplotlib style")

def jump_diffusion_euler(model, x0, t_span, dt, random_state=None):
    """
    Custom implementation of Euler-Maruyama for Jump Diffusion model.
    
    Args:
        model: The Jump Diffusion model
        x0: Initial price
        t_span: Time span
        dt: Time step
        random_state: Random seed
        
    Returns:
        tuple: (t, x) where t is time points and x is prices
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    
    # Initialize arrays for time and state
    t = np.linspace(t_start, t_end, n_steps)
    x = np.zeros(n_steps)
    x[0] = x0
    
    # Generate jumps for the entire period
    jump_times, jump_sizes = model.generate_jumps(t_span, dt)
    
    # Euler-Maruyama iteration with jumps
    for i in range(1, n_steps):
        t_current = t[i-1]
        x_current = x[i-1]
        
        # Generate random normal increment for Brownian motion
        dw = np.random.normal(0, np.sqrt(dt))
        
        # Standard Euler-Maruyama update for continuous part
        x_next = x_current + model.drift(x_current, t_current) * dt + \
                model.diffusion(x_current, t_current) * dw
        
        # Check if a jump occurs between t_current and t[i]
        for j, jump_time in enumerate(jump_times):
            if t_current <= jump_time < t[i]:
                # Apply the jump effect
                x_next = x_next * (1 + jump_sizes[j])
        
        x[i] = x_next
        
    return t, x

def generate_jump_paths(model, x0, t_span, dt, n_paths=1, random_state=None):
    """
    Generate multiple jump diffusion paths.
    
    Args:
        model: The Jump Diffusion model
        x0: Initial price
        t_span: Time span
        dt: Time step
        n_paths: Number of paths to generate
        random_state: Random seed
        
    Returns:
        tuple: (t, x) where t is time points and x is a 2D array of prices
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    
    # Initialize time array and paths matrix
    t = np.linspace(t_start, t_end, n_steps)
    x = np.zeros((n_paths, n_steps))
    
    # Generate n_paths simulation paths
    for i in range(n_paths):
        # Use different random seed for each path
        path_seed = None if random_state is None else random_state + i
        _, x[i, :] = jump_diffusion_euler(model, x0, t_span, dt, random_state=path_seed)
    
    return t, x

# Main simulation
if __name__ == "__main__":
    # Define plots directory
    PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots', 'jump_diffusion')
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Define parameters for the jump diffusion model
    mu = 0.10  # 10% annual drift
    sigma = 0.20  # 20% annual volatility
    lambda_j = 5.0  # Expected 5 jumps per year
    jump_mean = -0.05  # Mean jump size of -5% (log scale)
    jump_std = 0.10  # Jump size standard deviation
    s0 = 100.0  # Initial price $100
    
    # Print model parameters
    print("Merton Jump Diffusion Model Parameters:")
    print(f"Drift (μ): {mu:.2f} (expected annual return of {mu*100:.1f}%)")
    print(f"Volatility (σ): {sigma:.2f} (annual volatility of {sigma*100:.1f}%)")
    print(f"Jump intensity (λ): {lambda_j:.2f} (expected {lambda_j:.1f} jumps per year)")
    print(f"Mean jump size: {np.exp(jump_mean)-1:.2%} ({jump_mean:.4f} in log scale)")
    print(f"Jump size standard deviation: {jump_std:.2f}")
    print(f"Initial price (S₀): ${s0:.2f}")
    
    # Create model
    jump_diffusion_model = MertonJumpDiffusion(mu, sigma, lambda_j, jump_mean, jump_std, s0)
    
    # Simulation parameters
    t_span = (0.0, 1.0)  # 1 year
    dt = 1/252  # Daily (252 trading days per year)
    
    # 1. Simulate a single path with jumps
    print("\n1. Simulating a single path...")
    
    # Generate standard GBM path without jumps
    t_points, path_gbm = euler_maruyama(jump_diffusion_model, s0, t_span, dt, random_state=42)
    
    # Generate path with jumps
    t_points, path_jumps = jump_diffusion_euler(jump_diffusion_model, s0, t_span, dt, random_state=42)
    
    # Plot single path
    plt.figure(figsize=(12, 6))
    plt.plot(t_points, path_gbm, 'b-', label='Standard GBM (no jumps)')
    plt.plot(t_points, path_jumps, 'r-', label='Jump Diffusion')
    
    # Find jump points by identifying large price changes
    returns = np.diff(path_jumps) / path_jumps[:-1]
    jump_indices = np.where(np.abs(returns) > 0.02)[0]  # Threshold for jump detection
    
    # Mark jumps on the plot
    for idx in jump_indices:
        plt.axvline(t_points[idx], color='g', linestyle=':', alpha=0.3)
        plt.annotate(f'Jump: {returns[idx]:.1%}', 
                     xy=(t_points[idx], path_jumps[idx]), 
                     xytext=(t_points[idx], path_jumps[idx] * (1.1 if returns[idx] > 0 else 0.9)),
                     arrowprops=dict(arrowstyle='->'),
                     fontsize=8)
    
    plt.xlabel('Time (years)')
    plt.ylabel('Asset Price ($)')
    plt.title('Jump Diffusion: Single Path Simulation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, 'jump_diffusion_single_path.png'))
    print("Saved single path plot to 'plots/jump_diffusion/jump_diffusion_single_path.png'")
    
    # 2. Simulate multiple paths
    print("\n2. Simulating multiple paths...")
    n_paths = 50
    
    # Generate paths
    t_points, paths = generate_jump_paths(jump_diffusion_model, s0, t_span, dt, n_paths=n_paths, random_state=42)
    
    # Plot
    plt.figure(figsize=(12, 6))
    for i in range(n_paths):
        plt.plot(t_points, paths[i, :], 'b-', alpha=0.2)
    
    # Add mean path
    mean_path = np.mean(paths, axis=0)
    plt.plot(t_points, mean_path, 'r-', linewidth=2, label='Mean Path')
    
    # Add reference to initial price
    plt.axhline(s0, color='k', linestyle=':', label=f'Initial price (${s0:.2f})')
    
    plt.xlabel('Time (years)')
    plt.ylabel('Asset Price ($)')
    plt.title('Jump Diffusion: Multiple Paths Simulation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, 'jump_diffusion_multiple_paths.png'))
    print("Saved multiple paths plot to 'plots/jump_diffusion/jump_diffusion_multiple_paths.png'")
    
    # 3. Analyze distribution of final prices
    print("\n3. Analyzing distribution of final prices...")
    
    # Increase number of paths for better distribution estimation
    n_paths_dist = 5000
    t_points, paths = generate_jump_paths(jump_diffusion_model, s0, t_span, dt, n_paths=n_paths_dist, random_state=42)
    
    # Extract final prices
    final_prices = paths[:, -1]
    
    # Calculate statistics
    mean_price = np.mean(final_prices)
    std_price = np.std(final_prices)
    percentile_5 = np.percentile(final_prices, 5)
    percentile_95 = np.percentile(final_prices, 95)
    skewness = stats.skew(final_prices)
    kurtosis = stats.kurtosis(final_prices)
    
    print(f"Distribution of final prices (after 1 year):")
    print(f"Mean: ${mean_price:.2f}")
    print(f"Standard Deviation: ${std_price:.2f}")
    print(f"5th Percentile: ${percentile_5:.2f}")
    print(f"95th Percentile: ${percentile_95:.2f}")
    print(f"Skewness: {skewness:.4f}")
    print(f"Excess Kurtosis: {kurtosis:.4f}")
    
    # Plot histogram
    plt.figure(figsize=(12, 6))
    
    # Histogram of final prices
    n, bins, _ = plt.hist(final_prices, bins=50, density=True, alpha=0.7, 
                      label='Simulated Distribution')
    
    # For comparison, add normal distribution with same mean and std
    x = np.linspace(min(bins), max(bins), 100)
    plt.plot(x, stats.norm.pdf(x, mean_price, std_price), 'r-', 
             label='Normal Distribution\n(same mean & std)')
    
    # Add vertical lines for key statistics
    plt.axvline(mean_price, color='r', linestyle='-', label=f'Mean: ${mean_price:.2f}')
    plt.axvline(percentile_5, color='g', linestyle='--', label=f'5th Percentile: ${percentile_5:.2f}')
    plt.axvline(percentile_95, color='g', linestyle='--', label=f'95th Percentile: ${percentile_95:.2f}')
    
    plt.xlabel('Final Price ($)')
    plt.ylabel('Probability Density')
    plt.title('Jump Diffusion: Distribution of Final Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, 'jump_diffusion_distribution.png'))
    print("Saved distribution plot to 'plots/jump_diffusion/jump_diffusion_distribution.png'")
    
    # 4. Compare with different jump parameters
    print("\n4. Comparing different jump parameters...")
    
    # Define different jump parameters
    jump_params = [
        {"lambda": 1.0, "mean": -0.05, "std": 0.10, "label": "Rare Jumps (λ=1)"},
        {"lambda": 10.0, "mean": -0.02, "std": 0.05, "label": "Frequent Small Jumps (λ=10)"},
        {"lambda": 3.0, "mean": -0.10, "std": 0.15, "label": "Medium Frequency Large Jumps (λ=3)"},
        {"lambda": 5.0, "mean": 0.03, "std": 0.08, "label": "Positive Jumps (λ=5)"}
    ]
    
    # Create paths with different parameters
    plt.figure(figsize=(12, 6))
    
    # Add standard GBM for comparison
    t_points, path_gbm = euler_maruyama(jump_diffusion_model, s0, t_span, dt, random_state=42)
    plt.plot(t_points, path_gbm, 'k--', label='Standard GBM (no jumps)')
    
    for params in jump_params:
        # Create model with these parameters
        model = MertonJumpDiffusion(mu, sigma, params["lambda"], params["mean"], params["std"], s0)
        
        # Generate path
        _, path = jump_diffusion_euler(model, s0, t_span, dt, random_state=42)
        
        # Plot
        plt.plot(t_points, path, '-', label=params["label"])
    
    plt.xlabel('Time (years)')
    plt.ylabel('Asset Price ($)')
    plt.title('Jump Diffusion: Comparison of Different Jump Parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, 'jump_diffusion_comparison.png'))
    print("Saved comparison plot to 'plots/jump_diffusion/jump_diffusion_comparison.png'")
    
    # 5. Option pricing simulation 
    print("\n5. Simulating option prices...")

    # Option parameters
    strike_prices = np.linspace(80, 120, 9)  # Range of strike prices
    maturity = 1.0  # 1 year
    rfr = 0.05  # Risk-free rate
    
    # Calculate Black-Scholes and Jump Diffusion option prices
    # Risk-neutral parameters for jump diffusion
    n_paths_option = 10000
    
    # Function to calculate option payoffs
    def call_payoff(S, K):
        return np.maximum(S - K, 0)
    
    def put_payoff(S, K):
        return np.maximum(K - S, 0)
    
    # Black-Scholes formula for comparison
    def black_scholes_call(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    
    def black_scholes_put(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    
    # Calculate option prices
    call_prices_bs = np.array([black_scholes_call(s0, K, maturity, rfr, sigma) for K in strike_prices])
    put_prices_bs = np.array([black_scholes_put(s0, K, maturity, rfr, sigma) for K in strike_prices])
    
    # Calculate option prices using Monte Carlo for Jump Diffusion
    # Generate paths for risk-neutral pricing
    rn_model = MertonJumpDiffusion(rfr, sigma, lambda_j, jump_mean, jump_std, s0)
    t_points, paths = generate_jump_paths(rn_model, s0, (0, maturity), dt, n_paths=n_paths_option, random_state=42)
    
    # Calculate option prices
    call_prices_jd = np.zeros_like(strike_prices)
    put_prices_jd = np.zeros_like(strike_prices)
    
    # Get final prices
    final_prices = paths[:, -1]
    
    for i, K in enumerate(strike_prices):
        # Calculate discounted expected payoff
        call_prices_jd[i] = np.exp(-rfr * maturity) * np.mean(call_payoff(final_prices, K))
        put_prices_jd[i] = np.exp(-rfr * maturity) * np.mean(put_payoff(final_prices, K))
    
    # Plot option prices
    plt.figure(figsize=(12, 6))
    
    # Plot call options
    plt.subplot(1, 2, 1)
    plt.plot(strike_prices, call_prices_bs, 'b-', marker='o', label='Black-Scholes')
    plt.plot(strike_prices, call_prices_jd, 'r-', marker='x', label='Jump Diffusion')
    plt.xlabel('Strike Price ($)')
    plt.ylabel('Call Option Price ($)')
    plt.title('Call Option Prices')
    plt.axvline(s0, color='k', linestyle=':', label=f'Current Price (${s0:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot put options
    plt.subplot(1, 2, 2)
    plt.plot(strike_prices, put_prices_bs, 'b-', marker='o', label='Black-Scholes')
    plt.plot(strike_prices, put_prices_jd, 'r-', marker='x', label='Jump Diffusion')
    plt.xlabel('Strike Price ($)')
    plt.ylabel('Put Option Price ($)')
    plt.title('Put Option Prices')
    plt.axvline(s0, color='k', linestyle=':', label=f'Current Price (${s0:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'jump_diffusion_option_prices.png'))
    print("Saved option prices plot to 'plots/jump_diffusion/jump_diffusion_option_prices.png'")
    
    print("\nSimulation complete! All plots saved to 'plots/jump_diffusion/' directory.") 
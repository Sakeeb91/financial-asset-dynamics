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
from sde_asset_modeling.simulation.engine import SimulationEngine

try:
    from sde_asset_modeling.utils.plotting import setup_plotting_style
    # Set plotting style
    setup_plotting_style()
except ImportError:
    print("Plotting utilities not available, using default matplotlib style")

def jump_diffusion_euler_step(model, x, t, dt, dW, Z=None):
    """
    Custom Euler-Maruyama step for Jump Diffusion model that includes jumps.
    
    Args:
        model: The Jump Diffusion model
        x: Current price
        t: Current time
        dt: Time step
        dW: Wiener process increment
        Z: Additional random factor (unused)
        
    Returns:
        float: Next price
    """
    # Standard Euler drift and diffusion steps
    drift = model.drift(x, t) * dt
    diffusion = model.diffusion(x, t) * dW
    
    # Check if a jump occurs in this time step
    # Jump probability in time interval dt is lambda*dt
    jump_occurs = np.random.random() < model.lambda_j * dt
    
    if jump_occurs:
        # Generate jump size
        jump_size = np.exp(np.random.normal(model.jump_mean, model.jump_std)) - 1
        # Apply the jump
        jump_effect = x * jump_size
    else:
        jump_effect = 0
    
    # Combine continuous and jump components
    return x + drift + diffusion + jump_effect

# Main simulation function
if __name__ == "__main__":
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
    
    # Create simulation engine
    engine = SimulationEngine(jump_diffusion_model, s0, t_span, dt)
    
    # 1. Simulate a single path with jumps
    print("\n1. Simulating a single path...")
    
    # Generate jump times and sizes
    jump_times, jump_sizes = jump_diffusion_model.generate_jumps(t_span, dt)
    
    # Standard GBM path
    t_points, path_euler = engine.run_simulation(method='euler')
    
    # Path with custom jump diffusion step
    t_points, path_jumps = engine.run_simulation(step_func=jump_diffusion_euler_step)
    
    # Exact solution with jumps
    path_exact = jump_diffusion_model.exact_solution(t_points, np.zeros_like(t_points), jump_times=jump_times, jump_sizes=jump_sizes)
    
    # Print results
    print(f"Number of jumps: {len(jump_times)}")
    print(f"Final price (No jumps, Euler): ${path_euler[-1]:.2f}")
    print(f"Final price (With jumps, Euler): ${path_jumps[-1]:.2f}")
    print(f"Final price (Exact solution): ${path_exact[-1]:.2f}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(t_points, path_euler, 'b-', label='Standard GBM (no jumps)')
    plt.plot(t_points, path_jumps, 'r-', label='Jump Diffusion')
    
    # Mark jumps on the plot
    if len(jump_times) > 0:
        for i, (jt, js) in enumerate(zip(jump_times, jump_sizes)):
            # Find the closest time point
            idx = np.abs(t_points - jt).argmin()
            # Annotate the jump
            plt.axvline(jt, color='g', linestyle=':', alpha=0.3)
            plt.annotate(f'Jump {i+1}: {js:.1%}', 
                         xy=(jt, path_jumps[idx]), 
                         xytext=(jt, path_jumps[idx] * (1.1 if js > 0 else 0.9)),
                         arrowprops=dict(arrowstyle='->'),
                         fontsize=8)
    
    plt.xlabel('Time (years)')
    plt.ylabel('Asset Price ($)')
    plt.title('Jump Diffusion: Single Path Simulation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('jump_diffusion_single_path.png')
    print("Saved single path plot to 'jump_diffusion_single_path.png'")
    
    # 2. Simulate multiple paths
    print("\n2. Simulating multiple paths...")
    n_paths = 50
    
    # Create paths
    paths = np.zeros((n_paths, len(t_points)))
    
    for i in range(n_paths):
        # For each path, generate new jumps
        _, _, path = engine.simulate_path(step_func=jump_diffusion_euler_step)
        paths[i, :] = path
    
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
    plt.savefig('jump_diffusion_multiple_paths.png')
    print("Saved multiple paths plot to 'jump_diffusion_multiple_paths.png'")
    
    # 3. Analyze distribution of final prices
    print("\n3. Analyzing distribution of final prices...")
    
    # Increase number of paths for better distribution estimation
    n_paths_dist = 10000
    final_prices = np.zeros(n_paths_dist)
    
    for i in range(n_paths_dist):
        # Simulate path and get final price
        _, _, path = engine.simulate_path(step_func=jump_diffusion_euler_step)
        final_prices[i] = path[-1]
    
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
    plt.savefig('jump_diffusion_distribution.png')
    print("Saved distribution plot to 'jump_diffusion_distribution.png'")
    
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
    t_points, _, path_gbm = engine.simulate_path(method='euler')
    plt.plot(t_points, path_gbm, 'k--', label='Standard GBM (no jumps)')
    
    for params in jump_params:
        # Create model with these parameters
        model = MertonJumpDiffusion(mu, sigma, params["lambda"], params["mean"], params["std"], s0)
        engine_temp = SimulationEngine(model, s0, t_span, dt)
        
        # Generate path
        def custom_step(model, x, t, dt, dW, Z=None):
            # Standard Euler drift and diffusion steps
            drift = model.drift(x, t) * dt
            diffusion = model.diffusion(x, t) * dW
            
            # Check for jump
            jump_occurs = np.random.random() < model.lambda_j * dt
            
            if jump_occurs:
                jump_size = np.exp(np.random.normal(model.jump_mean, model.jump_std)) - 1
                jump_effect = x * jump_size
            else:
                jump_effect = 0
            
            return x + drift + diffusion + jump_effect
        
        _, _, path = engine_temp.simulate_path(step_func=custom_step)
        
        # Plot path
        plt.plot(t_points, path, label=params["label"])
    
    plt.xlabel('Time (years)')
    plt.ylabel('Asset Price ($)')
    plt.title('Jump Diffusion: Comparison of Jump Parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('jump_diffusion_comparison.png')
    print("Saved comparison plot to 'jump_diffusion_comparison.png'")
    
    # 5. Calculate option prices under jump diffusion
    print("\n5. Calculating option prices under jump diffusion...")
    
    # Define option parameters
    strike_prices = np.linspace(80, 120, 9)  # Strikes from $80 to $120
    rfr = 0.03  # 3% risk-free rate
    
    # Calculate option prices for different strikes
    n_options = 10000  # Number of paths for Monte Carlo
    call_prices = np.zeros(len(strike_prices))
    put_prices = np.zeros(len(strike_prices))
    
    # Create a risk-neutral model for option pricing
    # Adjust drift to risk-free rate
    rn_model = MertonJumpDiffusion(rfr, sigma, lambda_j, jump_mean, jump_std, s0)
    rn_engine = SimulationEngine(rn_model, s0, t_span, dt)
    
    # Generate paths
    final_prices = np.zeros(n_options)
    for i in range(n_options):
        _, _, path = rn_engine.simulate_path(step_func=jump_diffusion_euler_step)
        final_prices[i] = path[-1]
    
    # Calculate option prices
    discount_factor = np.exp(-rfr * (t_span[1] - t_span[0]))
    
    for i, K in enumerate(strike_prices):
        # Call option payoffs: max(S_T - K, 0)
        call_payoffs = np.maximum(final_prices - K, 0)
        call_prices[i] = np.mean(call_payoffs) * discount_factor
        
        # Put option payoffs: max(K - S_T, 0)
        put_payoffs = np.maximum(K - final_prices, 0)
        put_prices[i] = np.mean(put_payoffs) * discount_factor
    
    # Calculate Black-Scholes prices for comparison
    from scipy.stats import norm
    
    def black_scholes_call(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    def black_scholes_put(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    bs_call_prices = np.array([black_scholes_call(s0, K, t_span[1]-t_span[0], rfr, sigma) for K in strike_prices])
    bs_put_prices = np.array([black_scholes_put(s0, K, t_span[1]-t_span[0], rfr, sigma) for K in strike_prices])
    
    # Plot option prices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Call options
    ax1.plot(strike_prices, call_prices, 'bo-', label='Jump Diffusion')
    ax1.plot(strike_prices, bs_call_prices, 'ro--', label='Black-Scholes')
    ax1.set_xlabel('Strike Price ($)')
    ax1.set_ylabel('Option Price ($)')
    ax1.set_title('Call Option Prices')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Put options
    ax2.plot(strike_prices, put_prices, 'bo-', label='Jump Diffusion')
    ax2.plot(strike_prices, bs_put_prices, 'ro--', label='Black-Scholes')
    ax2.set_xlabel('Strike Price ($)')
    ax2.set_ylabel('Option Price ($)')
    ax2.set_title('Put Option Prices')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('jump_diffusion_option_prices.png')
    print("Saved option prices plot to 'jump_diffusion_option_prices.png'")
    
    # Print some option prices
    print(f"\nOption Prices (1-year maturity, S₀=${s0:.2f}, r={rfr:.1%}):")
    print(f"{'Strike':^10}|{'BS Call':^12}|{'JD Call':^12}|{'BS Put':^12}|{'JD Put':^12}")
    print("-" * 60)
    
    for i, K in enumerate(strike_prices):
        print(f"${K:^8.2f}|${bs_call_prices[i]:^10.2f}|${call_prices[i]:^10.2f}|${bs_put_prices[i]:^10.2f}|${put_prices[i]:^10.2f}")
    
    print("\nSimulation complete! All plots saved to the current directory.") 
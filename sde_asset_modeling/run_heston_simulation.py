#!/usr/bin/env python
"""
Heston Stochastic Volatility Simulation Script
This script demonstrates simulation of asset prices using the Heston stochastic volatility model.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import stats

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules from the project
from sde_asset_modeling.models.heston import HestonModel
from sde_asset_modeling.simulation.engine import SimulationEngine

try:
    from sde_asset_modeling.utils.plotting import setup_plotting_style
    # Set plotting style
    setup_plotting_style()
except ImportError:
    print("Plotting utilities not available, using default matplotlib style")

def heston_euler_step(model, S_v, t, dt, dW, Z=None):
    """
    Custom Euler-Maruyama step for Heston model.
    
    Args:
        model: The Heston model
        S_v: Tuple of (price, variance) at current time
        t: Current time
        dt: Time step
        dW: Tuple of (dW_S, dW_v) Wiener process increments
        Z: Additional random factor (unused)
        
    Returns:
        tuple: (S_next, v_next) next price and variance values
    """
    S, v = S_v
    dW_S, dW_v = dW
    
    # Ensure variance is positive
    v = max(1e-10, v)
    
    # Price step
    S_drift = model.mu * S * dt
    S_diffusion = np.sqrt(v) * S * dW_S
    S_next = S + S_drift + S_diffusion
    
    # Variance step
    v_drift = model.kappa * (model.theta - v) * dt
    v_diffusion = model.xi * np.sqrt(v) * dW_v
    v_next = v + v_drift + v_diffusion
    
    # Ensure variance remains positive
    v_next = max(1e-10, v_next)
    
    return (S_next, v_next)

def simulate_heston_path(model, t_span, dt, seed=None):
    """
    Simulate a single path of the Heston model.
    
    Args:
        model: The Heston model
        t_span: (t_start, t_end) time interval
        dt: Time step size
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (t_points, S_path, v_path) time points, price path, and variance path
    """
    if seed is not None:
        np.random.seed(seed)
    
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t_points = np.linspace(t_start, t_end, n_steps)
    
    # Initialize paths
    S_path = np.zeros(n_steps)
    v_path = np.zeros(n_steps)
    
    # Set initial values
    S_path[0] = model.x0
    v_path[0] = model.v0
    
    # Generate correlated Wiener processes
    W_S, W_v = model.generate_correlated_wiener_processes(n_steps-1, dt, n_paths=1)
    
    # Simulate path
    for i in range(1, n_steps):
        S_path[i], v_path[i] = heston_euler_step(
            model, 
            (S_path[i-1], v_path[i-1]), 
            t_points[i-1], 
            dt, 
            (W_S[0, i-1], W_v[0, i-1])
        )
    
    return t_points, S_path, v_path

def simulate_multiple_heston_paths(model, t_span, dt, n_paths=10):
    """
    Simulate multiple paths of the Heston model.
    
    Args:
        model: The Heston model
        t_span: (t_start, t_end) time interval
        dt: Time step size
        n_paths: Number of paths to simulate
        
    Returns:
        tuple: (t_points, S_paths, v_paths) time points, price paths, and variance paths
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t_points = np.linspace(t_start, t_end, n_steps)
    
    # Initialize paths
    S_paths = np.zeros((n_paths, n_steps))
    v_paths = np.zeros((n_paths, n_steps))
    
    # Generate correlated Wiener processes for all paths
    W_S, W_v = model.generate_correlated_wiener_processes(n_steps-1, dt, n_paths=n_paths)
    
    # Set initial values
    S_paths[:, 0] = model.x0
    v_paths[:, 0] = model.v0
    
    # Simulate paths
    for path_idx in range(n_paths):
        for i in range(1, n_steps):
            S_paths[path_idx, i], v_paths[path_idx, i] = heston_euler_step(
                model, 
                (S_paths[path_idx, i-1], v_paths[path_idx, i-1]), 
                t_points[i-1], 
                dt, 
                (W_S[path_idx, i-1], W_v[path_idx, i-1])
            )
    
    return t_points, S_paths, v_paths

# Main simulation function
if __name__ == "__main__":
    # Define parameters for the Heston model
    mu = 0.10  # 10% annual drift
    kappa = 2.0  # Mean reversion speed
    theta = 0.04  # Long-term variance (corresponds to 20% volatility)
    xi = 0.3  # Volatility of volatility
    rho = -0.7  # Correlation between price and volatility (-0.7 is typical)
    s0 = 100.0  # Initial price $100
    v0 = 0.04  # Initial variance (corresponds to 20% volatility)
    
    # Print model parameters
    print("Heston Stochastic Volatility Model Parameters:")
    print(f"Drift (μ): {mu:.2f} (expected annual return of {mu*100:.1f}%)")
    print(f"Mean reversion speed (κ): {kappa:.2f}")
    print(f"Long-term variance (θ): {theta:.4f} (volatility of {np.sqrt(theta)*100:.1f}%)")
    print(f"Volatility of volatility (ξ): {xi:.2f}")
    print(f"Price-volatility correlation (ρ): {rho:.2f}")
    print(f"Initial price (S₀): ${s0:.2f}")
    print(f"Initial variance (v₀): {v0:.4f} (volatility of {np.sqrt(v0)*100:.1f}%)")
    
    # Create model
    heston_model = HestonModel(mu, kappa, theta, xi, rho, s0, v0)
    
    # Simulation parameters
    t_span = (0.0, 1.0)  # 1 year
    dt = 1/252  # Daily (252 trading days per year)
    
    # 1. Simulate a single path
    print("\n1. Simulating a single path...")
    
    # Set a seed for reproducibility
    t_points, price_path, var_path = simulate_heston_path(heston_model, t_span, dt, seed=42)
    vol_path = np.sqrt(var_path)  # Convert variance to volatility for plotting
    
    # Print results
    print(f"Final price: ${price_path[-1]:.2f}")
    print(f"Final volatility: {vol_path[-1]*100:.2f}%")
    
    # Plot price and volatility paths
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    
    # Price path
    ax1.plot(t_points, price_path, 'b-')
    ax1.set_ylabel('Asset Price ($)')
    ax1.set_title('Heston Model: Single Path - Price')
    ax1.grid(True, alpha=0.3)
    
    # Volatility path
    ax2.plot(t_points, vol_path * 100, 'r-')  # Multiply by 100 to show in percentage
    ax2.axhline(np.sqrt(theta) * 100, color='g', linestyle=':', 
               label=f'Long-term volatility ({np.sqrt(theta)*100:.1f}%)')
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Volatility (%)')
    ax2.set_title('Heston Model: Single Path - Volatility')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heston_single_path.png')
    print("Saved single path plot to 'heston_single_path.png'")
    
    # 2. Simulate multiple paths
    print("\n2. Simulating multiple paths...")
    n_paths = 50
    
    t_points, price_paths, var_paths = simulate_multiple_heston_paths(
        heston_model, t_span, dt, n_paths=n_paths
    )
    vol_paths = np.sqrt(var_paths)  # Convert variance to volatility
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    
    # Price paths
    for i in range(n_paths):
        ax1.plot(t_points, price_paths[i, :], 'b-', alpha=0.2)
    
    # Add mean price path
    mean_price_path = np.mean(price_paths, axis=0)
    ax1.plot(t_points, mean_price_path, 'r-', linewidth=2, label='Mean Path')
    
    # Add reference to initial price
    ax1.axhline(s0, color='k', linestyle=':', label=f'Initial price (${s0:.2f})')
    
    ax1.set_ylabel('Asset Price ($)')
    ax1.set_title('Heston Model: Multiple Paths - Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volatility paths
    for i in range(n_paths):
        ax2.plot(t_points, vol_paths[i, :] * 100, 'r-', alpha=0.2)
    
    # Add mean volatility path
    mean_vol_path = np.mean(vol_paths, axis=0) * 100  # Percentage
    ax2.plot(t_points, mean_vol_path, 'b-', linewidth=2, label='Mean Volatility')
    
    # Add reference to long-term volatility
    ax2.axhline(np.sqrt(theta) * 100, color='g', linestyle=':', 
               label=f'Long-term volatility ({np.sqrt(theta)*100:.1f}%)')
    
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Volatility (%)')
    ax2.set_title('Heston Model: Multiple Paths - Volatility')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heston_multiple_paths.png')
    print("Saved multiple paths plot to 'heston_multiple_paths.png'")
    
    # 3. Analyze distribution of final prices
    print("\n3. Analyzing distribution of final prices...")
    
    # Increase number of paths for better distribution estimation
    n_paths_dist = 10000
    _, price_paths_dist, _ = simulate_multiple_heston_paths(
        heston_model, t_span, dt, n_paths=n_paths_dist
    )
    
    # Get final prices
    final_prices = price_paths_dist[:, -1]
    
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
    plt.title('Heston Model: Distribution of Final Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('heston_distribution.png')
    print("Saved distribution plot to 'heston_distribution.png'")
    
    # 4. Compare different volatility and correlation parameters
    print("\n4. Comparing different volatility parameters...")
    
    # Define parameter sets
    param_sets = [
        {"name": "Low Vol of Vol", "xi": 0.1, "rho": -0.7},
        {"name": "High Vol of Vol", "xi": 0.5, "rho": -0.7},
        {"name": "No Correlation", "xi": 0.3, "rho": 0.0},
        {"name": "Positive Correlation", "xi": 0.3, "rho": 0.5}
    ]
    
    # Simulate paths with different parameters
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    
    # Generate a single standard path for reference
    t_points, base_price_path, base_var_path = simulate_heston_path(
        heston_model, t_span, dt, seed=42
    )
    base_vol_path = np.sqrt(base_var_path)
    
    # Plot reference paths
    ax1.plot(t_points, base_price_path, 'k-', label='Base Model')
    ax2.plot(t_points, base_vol_path * 100, 'k-', label='Base Model')
    
    # Simulate paths with different parameters
    for params in param_sets:
        # Create model with these parameters
        temp_model = HestonModel(
            mu, kappa, theta, params["xi"], params["rho"], s0, v0
        )
        
        # Simulate a path
        _, price_path, var_path = simulate_heston_path(
            temp_model, t_span, dt, seed=42  # Same seed for comparison
        )
        vol_path = np.sqrt(var_path)
        
        # Plot
        ax1.plot(t_points, price_path, label=params["name"])
        ax2.plot(t_points, vol_path * 100, label=params["name"])
    
    # Format price plot
    ax1.set_ylabel('Asset Price ($)')
    ax1.set_title('Heston Model: Price Paths for Different Parameters')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Format volatility plot
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Volatility (%)')
    ax2.set_title('Heston Model: Volatility Paths for Different Parameters')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heston_parameter_comparison.png')
    print("Saved parameter comparison plot to 'heston_parameter_comparison.png'")
    
    # 5. Demonstrate volatility clustering
    print("\n5. Demonstrating volatility clustering...")
    
    # Simulate a longer time period to better observe clustering
    extended_t_span = (0.0, 5.0)  # 5 years
    
    # Create a model with higher vol of vol for more pronounced clustering
    cluster_model = HestonModel(mu, kappa, theta, 0.5, -0.7, s0, v0)
    
    # Simulate path
    t_points_ext, price_path_ext, var_path_ext = simulate_heston_path(
        cluster_model, extended_t_span, dt, seed=123
    )
    vol_path_ext = np.sqrt(var_path_ext)
    
    # Calculate returns
    returns = np.diff(np.log(price_path_ext))
    
    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Price path
    ax1.plot(t_points_ext, price_path_ext, 'b-')
    ax1.set_ylabel('Asset Price ($)')
    ax1.set_title('Heston Model: Price Path (5 Years)')
    ax1.grid(True, alpha=0.3)
    
    # Volatility path
    ax2.plot(t_points_ext, vol_path_ext * 100, 'r-')
    ax2.axhline(np.sqrt(theta) * 100, color='g', linestyle=':', 
               label=f'Long-term volatility ({np.sqrt(theta)*100:.1f}%)')
    ax2.set_ylabel('Volatility (%)')
    ax2.set_title('Heston Model: Volatility Path (5 Years)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Returns
    ax3.plot(t_points_ext[1:], returns, 'g-')
    ax3.set_xlabel('Time (years)')
    ax3.set_ylabel('Log Returns')
    ax3.set_title('Heston Model: Daily Log Returns (5 Years)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heston_volatility_clustering.png')
    print("Saved volatility clustering plot to 'heston_volatility_clustering.png'")
    
    # 6. Calculate option prices under Heston model
    print("\n6. Calculating option prices under Heston model...")
    
    # Define option parameters
    strike_prices = np.linspace(80, 120, 9)  # Strikes from $80 to $120
    rfr = 0.03  # 3% risk-free rate
    
    # Create a risk-neutral model for option pricing
    # Adjust drift to risk-free rate
    rn_model = HestonModel(rfr, kappa, theta, xi, rho, s0, v0)
    
    # Calculate option prices for different strikes
    n_options = 10000  # Number of paths for Monte Carlo
    _, price_paths_opt, _ = simulate_multiple_heston_paths(
        rn_model, t_span, dt, n_paths=n_options
    )
    
    # Get final prices
    final_prices = price_paths_opt[:, -1]
    
    # Calculate option prices
    discount_factor = np.exp(-rfr * (t_span[1] - t_span[0]))
    heston_call_prices = np.zeros(len(strike_prices))
    heston_put_prices = np.zeros(len(strike_prices))
    
    for i, K in enumerate(strike_prices):
        # Call option payoffs: max(S_T - K, 0)
        call_payoffs = np.maximum(final_prices - K, 0)
        heston_call_prices[i] = np.mean(call_payoffs) * discount_factor
        
        # Put option payoffs: max(K - S_T, 0)
        put_payoffs = np.maximum(K - final_prices, 0)
        heston_put_prices[i] = np.mean(put_payoffs) * discount_factor
    
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
    
    # Use long-term volatility for Black-Scholes
    bs_sigma = np.sqrt(theta)
    bs_call_prices = np.array([black_scholes_call(s0, K, t_span[1]-t_span[0], rfr, bs_sigma) 
                              for K in strike_prices])
    bs_put_prices = np.array([black_scholes_put(s0, K, t_span[1]-t_span[0], rfr, bs_sigma) 
                             for K in strike_prices])
    
    # Volatility smile/skew calculation
    implied_vols = np.zeros(len(strike_prices))
    
    # Simple algorithm to find implied volatility (bisection method)
    def find_implied_vol(S, K, T, r, price, option_type='call', tol=1e-5, max_iter=100):
        sigma_low = 0.001
        sigma_high = 1.0
        
        for i in range(max_iter):
            sigma_mid = (sigma_low + sigma_high) / 2.0
            
            if option_type == 'call':
                price_mid = black_scholes_call(S, K, T, r, sigma_mid)
            else:
                price_mid = black_scholes_put(S, K, T, r, sigma_mid)
            
            if abs(price_mid - price) < tol:
                return sigma_mid
            
            if price_mid > price:
                sigma_high = sigma_mid
            else:
                sigma_low = sigma_mid
        
        return (sigma_low + sigma_high) / 2.0
    
    # Calculate implied volatility for each strike
    for i, K in enumerate(strike_prices):
        implied_vols[i] = find_implied_vol(s0, K, t_span[1]-t_span[0], rfr, 
                                           heston_call_prices[i], 
                                           option_type='call')
    
    # Plot option prices and implied volatility
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Call option prices
    ax1.plot(strike_prices, heston_call_prices, 'bo-', label='Heston Model')
    ax1.plot(strike_prices, bs_call_prices, 'ro--', label='Black-Scholes')
    ax1.set_xlabel('Strike Price ($)')
    ax1.set_ylabel('Option Price ($)')
    ax1.set_title('Call Option Prices')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Put option prices
    ax2.plot(strike_prices, heston_put_prices, 'bo-', label='Heston Model')
    ax2.plot(strike_prices, bs_put_prices, 'ro--', label='Black-Scholes')
    ax2.set_xlabel('Strike Price ($)')
    ax2.set_ylabel('Option Price ($)')
    ax2.set_title('Put Option Prices')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Implied volatility smile/skew
    ax3.plot(strike_prices, implied_vols * 100, 'go-', label='Implied Volatility')
    ax3.axhline(np.sqrt(theta) * 100, color='k', linestyle=':', 
               label=f'Black-Scholes Volatility ({np.sqrt(theta)*100:.1f}%)')
    ax3.set_xlabel('Strike Price ($)')
    ax3.set_ylabel('Implied Volatility (%)')
    ax3.set_title('Volatility Smile/Skew')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heston_option_pricing.png')
    print("Saved option pricing and volatility smile plot to 'heston_option_pricing.png'")
    
    # Print some option prices
    print(f"\nOption Prices (1-year maturity, S₀=${s0:.2f}, r={rfr:.1%}):")
    print(f"{'Strike':^10}|{'BS Call':^12}|{'Heston Call':^12}|{'BS Put':^12}|{'Heston Put':^12}")
    print("-" * 60)
    
    for i, K in enumerate(strike_prices):
        print(f"${K:^8.2f}|${bs_call_prices[i]:^10.2f}|${heston_call_prices[i]:^10.2f}|${bs_put_prices[i]:^10.2f}|${heston_put_prices[i]:^10.2f}")
    
    print("\nSimulation complete! All plots saved to the current directory.") 
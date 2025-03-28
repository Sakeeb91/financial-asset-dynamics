#!/usr/bin/env python
"""
Correlated Assets Simulation Script
This script demonstrates simulation of correlated financial assets using GBM models.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from scipy import stats

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules from the project
from sde_asset_modeling.models.gbm import GeometricBrownianMotion
from sde_asset_modeling.simulation.simulators import euler_maruyama
from sde_asset_modeling.utils.data_fetcher import create_correlated_returns

try:
    from sde_asset_modeling.utils.plotting import setup_plotting_style
    # Set plotting style
    setup_plotting_style()
except ImportError:
    print("Plotting utilities not available, using default matplotlib style")

def simulate_correlated_gbm(mu_vector, sigma_vector, correlation_matrix, s0_vector, t_span, dt, n_paths=1):
    """
    Simulate correlated GBM processes.
    
    Args:
        mu_vector: Array of drift parameters for each asset
        sigma_vector: Array of volatility parameters for each asset
        correlation_matrix: Correlation matrix between assets
        s0_vector: Array of initial prices for each asset
        t_span: (t_start, t_end) time interval
        dt: Time step size
        n_paths: Number of simulation paths
    
    Returns:
        tuple: (t, paths) where t is an array of time points and paths is a 3D array 
               with shape (n_paths, n_assets, len(t))
    """
    n_assets = len(mu_vector)
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t = np.linspace(t_start, t_end, n_steps)
    
    # Create covariance matrix from correlation matrix and volatilities
    vols_matrix = np.diag(sigma_vector)
    covariance_matrix = vols_matrix @ correlation_matrix @ vols_matrix
    
    # Cholesky decomposition for generating correlated normal variables
    L = np.linalg.cholesky(correlation_matrix)
    
    # Initialize paths array: (n_paths, n_assets, n_steps)
    paths = np.zeros((n_paths, n_assets, n_steps))
    paths[:, :, 0] = s0_vector
    
    # Simulate paths
    for path_idx in range(n_paths):
        for i in range(1, n_steps):
            # Generate uncorrelated standard normal variables
            Z = np.random.standard_normal(n_assets)
            
            # Transform to correlated variables
            W = L @ Z
            
            # Scale by sqrt(dt) to get Brownian increments
            dW = W * np.sqrt(dt)
            
            # Apply Euler-Maruyama step for each asset
            for asset_idx in range(n_assets):
                mu = mu_vector[asset_idx]
                sigma = sigma_vector[asset_idx]
                S = paths[path_idx, asset_idx, i-1]
                
                # GBM step
                drift = mu * S * dt
                diffusion = sigma * S * dW[asset_idx]
                paths[path_idx, asset_idx, i] = S + drift + diffusion
    
    return t, paths

# Main simulation
if __name__ == "__main__":
    # Define parameters for 3 assets (e.g., stocks)
    asset_names = ['Asset A', 'Asset B', 'Asset C']
    n_assets = len(asset_names)
    
    # Parameters for each asset
    mu_vector = np.array([0.08, 0.12, 0.10])  # Annual expected returns: 8%, 12%, 10%
    sigma_vector = np.array([0.20, 0.30, 0.25])  # Annual volatilities: 20%, 30%, 25%
    s0_vector = np.array([100.0, 50.0, 75.0])  # Initial prices: $100, $50, $75
    
    # Correlation matrix (symmetric positive definite)
    correlation_matrix = np.array([
        [1.00, 0.60, 0.40],  # Asset A correlations
        [0.60, 1.00, 0.70],  # Asset B correlations
        [0.40, 0.70, 1.00]   # Asset C correlations
    ])
    
    # Print parameters
    print("Correlated Assets Simulation Parameters:")
    for i in range(n_assets):
        print(f"{asset_names[i]}:")
        print(f"  - Drift (μ): {mu_vector[i]:.2f} (expected annual return of {mu_vector[i]*100:.1f}%)")
        print(f"  - Volatility (σ): {sigma_vector[i]:.2f} (annual volatility of {sigma_vector[i]*100:.1f}%)")
        print(f"  - Initial price (S₀): ${s0_vector[i]:.2f}")
    
    print("\nCorrelation Matrix:")
    for i in range(n_assets):
        print(f"  {asset_names[i]}: " + " ".join([f"{correlation_matrix[i,j]:.2f}" for j in range(n_assets)]))
    
    # Simulation parameters
    t_span = (0.0, 1.0)  # 1 year
    dt = 1/252  # Daily timesteps (252 trading days per year)
    
    # 1. Simulate single path for each asset
    print("\n1. Simulating a single path for each asset...")
    t, paths_single = simulate_correlated_gbm(
        mu_vector, sigma_vector, correlation_matrix, s0_vector, t_span, dt, n_paths=1
    )
    
    # Plot single path
    plt.figure(figsize=(12, 6))
    for i in range(n_assets):
        plt.plot(t, paths_single[0, i, :], label=asset_names[i])
    plt.xlabel('Time (years)')
    plt.ylabel('Asset Price ($)')
    plt.title('Correlated Assets: Single Path Simulation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('correlated_single_path.png')
    print("Saved single path plot to 'correlated_single_path.png'")
    
    # 2. Simulate multiple paths
    print("\n2. Simulating multiple paths...")
    n_sim_paths = 50
    t, paths_multi = simulate_correlated_gbm(
        mu_vector, sigma_vector, correlation_matrix, s0_vector, t_span, dt, n_paths=n_sim_paths
    )
    
    # Plot multiple paths for each asset
    fig, axes = plt.subplots(n_assets, 1, figsize=(12, 9), sharex=True)
    for i in range(n_assets):
        for path_idx in range(n_sim_paths):
            axes[i].plot(t, paths_multi[path_idx, i, :], 'b-', alpha=0.2)
        
        # Add mean path
        mean_path = np.mean(paths_multi[:, i, :], axis=0)
        axes[i].plot(t, mean_path, 'r-', linewidth=2, label='Mean Path')
        
        # Horizontal line at initial price
        axes[i].axhline(s0_vector[i], color='k', linestyle=':', label=f'Initial price (${s0_vector[i]:.2f})')
        
        axes[i].set_ylabel(f'{asset_names[i]} Price ($)')
        axes[i].set_title(f'{asset_names[i]}: Multiple Paths')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (years)')
    plt.tight_layout()
    plt.savefig('correlated_multiple_paths.png')
    print("Saved multiple paths plot to 'correlated_multiple_paths.png'")
    
    # 3. Analyze correlations in simulated paths
    print("\n3. Analyzing correlations in simulated returns...")
    
    # Increase number of paths for better correlation estimation
    n_corr_paths = 1000
    t, paths_corr = simulate_correlated_gbm(
        mu_vector, sigma_vector, correlation_matrix, s0_vector, t_span, dt, n_paths=n_corr_paths
    )
    
    # Calculate returns for each asset
    returns = np.zeros((n_corr_paths, n_assets, len(t)-1))
    for i in range(n_assets):
        for path_idx in range(n_corr_paths):
            prices = paths_corr[path_idx, i, :]
            returns[path_idx, i, :] = np.diff(np.log(prices))
    
    # Calculate correlation matrix from simulated returns
    sim_corr_matrix = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            # Flatten returns across all paths and time steps
            returns_i = returns[:, i, :].flatten()
            returns_j = returns[:, j, :].flatten()
            sim_corr_matrix[i, j] = np.corrcoef(returns_i, returns_j)[0, 1]
    
    # Visualize correlation matrices (input vs. simulated)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Input correlation matrix
    im0 = axes[0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title('Input Correlation Matrix')
    axes[0].set_xticks(np.arange(n_assets))
    axes[0].set_yticks(np.arange(n_assets))
    axes[0].set_xticklabels(asset_names)
    axes[0].set_yticklabels(asset_names)
    
    # Add correlation values
    for i in range(n_assets):
        for j in range(n_assets):
            axes[0].text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                       ha='center', va='center', color='black')
    
    # Simulated correlation matrix
    im1 = axes[1].imshow(sim_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title('Simulated Correlation Matrix')
    axes[1].set_xticks(np.arange(n_assets))
    axes[1].set_yticks(np.arange(n_assets))
    axes[1].set_xticklabels(asset_names)
    axes[1].set_yticklabels(asset_names)
    
    # Add correlation values
    for i in range(n_assets):
        for j in range(n_assets):
            axes[1].text(j, i, f'{sim_corr_matrix[i, j]:.2f}', 
                       ha='center', va='center', color='black')
    
    # Add colorbar
    fig.colorbar(im1, ax=axes.ravel().tolist())
    
    plt.tight_layout()
    plt.savefig('correlated_correlations.png')
    print("Saved correlation analysis to 'correlated_correlations.png'")
    
    # 4. Analyze portfolio performance
    print("\n4. Analyzing portfolio performance...")
    
    # Define weights for a simple portfolio
    weights = np.array([0.4, 0.3, 0.3])  # 40% in Asset A, 30% in Assets B and C
    
    # Calculate portfolio values
    portfolio_values = np.zeros((n_corr_paths, len(t)))
    for path_idx in range(n_corr_paths):
        # Initial portfolio value
        initial_investment = np.sum(s0_vector * weights)
        
        # Calculate number of shares for each asset
        shares = weights * initial_investment / s0_vector
        
        # Calculate portfolio value at each time step
        for time_idx in range(len(t)):
            portfolio_values[path_idx, time_idx] = np.sum(
                paths_corr[path_idx, :, time_idx] * shares
            )
    
    # Plot portfolio paths
    plt.figure(figsize=(12, 6))
    
    # Plot individual portfolio paths
    for path_idx in range(min(n_corr_paths, 100)):  # Plot max 100 paths to avoid clutter
        plt.plot(t, portfolio_values[path_idx, :], 'b-', alpha=0.1)
    
    # Plot mean portfolio path
    mean_portfolio = np.mean(portfolio_values, axis=0)
    plt.plot(t, mean_portfolio, 'r-', linewidth=2, label='Mean Portfolio Value')
    
    # Plot initial portfolio value
    initial_portfolio = portfolio_values[0, 0]
    plt.axhline(initial_portfolio, color='k', linestyle=':', 
               label=f'Initial portfolio value (${initial_portfolio:.2f})')
    
    # Calculate 5% and 95% percentiles
    percentile_5 = np.percentile(portfolio_values, 5, axis=0)
    percentile_95 = np.percentile(portfolio_values, 95, axis=0)
    
    # Plot confidence interval
    plt.fill_between(t, percentile_5, percentile_95, alpha=0.2, color='g', 
                    label='90% Confidence Interval')
    
    plt.xlabel('Time (years)')
    plt.ylabel('Portfolio Value ($)')
    plt.title(f'Portfolio Performance (Weights: {weights[0]:.1f}/{weights[1]:.1f}/{weights[2]:.1f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('correlated_portfolio.png')
    print("Saved portfolio analysis to 'correlated_portfolio.png'")
    
    # Calculate statistics for final portfolio value
    final_portfolio_values = portfolio_values[:, -1]
    mean_final = np.mean(final_portfolio_values)
    std_final = np.std(final_portfolio_values)
    median_final = np.median(final_portfolio_values)
    min_final = np.min(final_portfolio_values)
    max_final = np.max(final_portfolio_values)
    percentile_5_final = np.percentile(final_portfolio_values, 5)
    percentile_95_final = np.percentile(final_portfolio_values, 95)
    
    # Print portfolio statistics
    print("\nPortfolio Statistics (after 1 year):")
    print(f"Initial Value: ${initial_portfolio:.2f}")
    print(f"Mean Final Value: ${mean_final:.2f}")
    print(f"Mean Return: {(mean_final/initial_portfolio - 1)*100:.2f}%")
    print(f"Standard Deviation: ${std_final:.2f}")
    print(f"Median Final Value: ${median_final:.2f}")
    print(f"Minimum Final Value: ${min_final:.2f}")
    print(f"Maximum Final Value: ${max_final:.2f}")
    print(f"5% VaR: ${initial_portfolio - percentile_5_final:.2f}")
    print(f"90% Confidence Interval: [${percentile_5_final:.2f}, ${percentile_95_final:.2f}]")
    
    # Plot histogram of final portfolio values
    plt.figure(figsize=(12, 6))
    plt.hist(final_portfolio_values, bins=50, alpha=0.7, density=True)
    plt.axvline(initial_portfolio, color='k', linestyle=':', 
               label=f'Initial: ${initial_portfolio:.2f}')
    plt.axvline(mean_final, color='r', linestyle='-', 
               label=f'Mean Final: ${mean_final:.2f}')
    plt.axvline(percentile_5_final, color='g', linestyle='--', 
               label=f'5th Percentile: ${percentile_5_final:.2f}')
    plt.axvline(percentile_95_final, color='g', linestyle='--', 
               label=f'95th Percentile: ${percentile_95_final:.2f}')
    
    plt.xlabel('Final Portfolio Value ($)')
    plt.ylabel('Probability Density')
    plt.title('Distribution of Final Portfolio Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('correlated_portfolio_distribution.png')
    print("Saved portfolio distribution to 'correlated_portfolio_distribution.png'")
    
    print("\nSimulation complete! All plots saved to the current directory.") 
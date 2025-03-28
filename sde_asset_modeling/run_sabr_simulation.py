#!/usr/bin/env python
"""
SABR Model Simulation Script
This script demonstrates simulation of forward rates using the SABR model,
commonly used for interest rate derivatives and volatility smile modeling.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import stats

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules from the project
from sde_asset_modeling.models.sabr import SABRModel
from sde_asset_modeling.simulation.engine import SimulationEngine

try:
    from sde_asset_modeling.utils.plotting import setup_plotting_style
    # Set plotting style
    setup_plotting_style()
except ImportError:
    print("Plotting utilities not available, using default matplotlib style")

def sabr_euler_step(model, F_alpha, t, dt, dW, Z=None):
    """
    Custom Euler-Maruyama step for SABR model.
    
    Args:
        model: The SABR model
        F_alpha: Tuple of (forward_rate, alpha) at current time
        t: Current time
        dt: Time step
        dW: Tuple of (dW_F, dW_alpha) Wiener process increments
        Z: Additional random factor (unused)
        
    Returns:
        tuple: (F_next, alpha_next) next forward rate and volatility values
    """
    F, alpha = F_alpha
    dW_F, dW_alpha = dW
    
    # Forward rate step (no drift in risk-neutral measure)
    F_diffusion = alpha * (abs(F)**model.beta) * dW_F
    F_next = F + F_diffusion
    
    # Volatility step (log-normal volatility)
    alpha_diffusion = model.nu * alpha * dW_alpha
    alpha_next = alpha + alpha_diffusion
    
    # Ensure positive values (optional but recommended for numerical stability)
    F_next = max(1e-6, F_next)  # Prevent negative rates for simplicity
    alpha_next = max(1e-6, alpha_next)  # Prevent negative volatility
    
    return (F_next, alpha_next)

def simulate_sabr_path(model, t_span, dt, seed=None):
    """
    Simulate a single path of the SABR model.
    
    Args:
        model: The SABR model
        t_span: (t_start, t_end) time interval
        dt: Time step size
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (t_points, F_path, alpha_path) time points, forward rate path, and volatility path
    """
    if seed is not None:
        np.random.seed(seed)
    
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t_points = np.linspace(t_start, t_end, n_steps)
    
    # Initialize paths
    F_path = np.zeros(n_steps)
    alpha_path = np.zeros(n_steps)
    
    # Set initial values
    F_path[0] = model.x0
    alpha_path[0] = model.alpha0
    
    # Generate correlated Wiener processes
    W_F, W_alpha = model.generate_correlated_wiener_processes(n_steps-1, dt, n_paths=1)
    
    # Simulate path
    for i in range(1, n_steps):
        F_path[i], alpha_path[i] = sabr_euler_step(
            model, 
            (F_path[i-1], alpha_path[i-1]), 
            t_points[i-1], 
            dt, 
            (W_F[0, i-1], W_alpha[0, i-1])
        )
    
    return t_points, F_path, alpha_path

def simulate_multiple_sabr_paths(model, t_span, dt, n_paths=10):
    """
    Simulate multiple paths of the SABR model.
    
    Args:
        model: The SABR model
        t_span: (t_start, t_end) time interval
        dt: Time step size
        n_paths: Number of paths to simulate
        
    Returns:
        tuple: (t_points, F_paths, alpha_paths) time points, forward rate paths, and volatility paths
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t_points = np.linspace(t_start, t_end, n_steps)
    
    # Initialize paths
    F_paths = np.zeros((n_paths, n_steps))
    alpha_paths = np.zeros((n_paths, n_steps))
    
    # Generate correlated Wiener processes for all paths
    W_F, W_alpha = model.generate_correlated_wiener_processes(n_steps-1, dt, n_paths=n_paths)
    
    # Set initial values
    F_paths[:, 0] = model.x0
    alpha_paths[:, 0] = model.alpha0
    
    # Simulate paths
    for path_idx in range(n_paths):
        for i in range(1, n_steps):
            F_paths[path_idx, i], alpha_paths[path_idx, i] = sabr_euler_step(
                model, 
                (F_paths[path_idx, i-1], alpha_paths[path_idx, i-1]), 
                t_points[i-1], 
                dt, 
                (W_F[path_idx, i-1], W_alpha[path_idx, i-1])
            )
    
    return t_points, F_paths, alpha_paths

def calculate_black_price(F, K, T, sigma, is_call=True):
    """
    Calculate Black model price for a call or put option.
    
    Args:
        F: Forward rate
        K: Strike rate
        T: Time to expiry
        sigma: Volatility
        is_call: True for call option, False for put option
        
    Returns:
        float: Option price in Black model
    """
    d1 = (np.log(F/K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if is_call:
        return (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
    else:
        return (K * stats.norm.cdf(-d2) - F * stats.norm.cdf(-d1))

# Main simulation function
if __name__ == "__main__":
    # Define plots directory
    PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots', 'sabr')
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Define parameters for the SABR model
    # This example uses parameters typical for interest rate caps/floors
    alpha = 0.15  # Initial volatility level
    beta = 0.5  # CEV parameter (0.5 common for interest rates)
    nu = 0.4  # Volatility of volatility
    rho = -0.3  # Correlation between rate and volatility
    f0 = 0.03  # Initial forward rate (3%)
    alpha0 = alpha  # Initial volatility
    
    # Print model parameters
    print("SABR Model Parameters:")
    print(f"Initial forward rate (F₀): {f0:.2%}")
    print(f"Initial volatility (α₀): {alpha0:.2f}")
    print(f"CEV parameter (β): {beta:.2f}")
    print(f"Volatility of volatility (ν): {nu:.2f}")
    print(f"Correlation (ρ): {rho:.2f}")
    
    # Create model
    sabr_model = SABRModel(alpha, beta, nu, rho, f0, alpha0)
    
    # Simulation parameters
    t_span = (0.0, 1.0)  # 1 year
    dt = 1/252  # Daily (252 trading days per year)
    
    # 1. Simulate a single path
    print("\n1. Simulating a single path...")
    
    # Set a seed for reproducibility
    t_points, forward_path, alpha_path = simulate_sabr_path(sabr_model, t_span, dt, seed=42)
    
    # Print results
    print(f"Initial forward rate: {f0:.2%}")
    print(f"Final forward rate: {forward_path[-1]:.2%}")
    print(f"Initial volatility: {alpha0:.2f}")
    print(f"Final volatility: {alpha_path[-1]:.2f}")
    
    # Plot forward rate and volatility paths
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    
    # Forward rate path
    ax1.plot(t_points, forward_path * 100, 'b-')  # *100 to show in percentage
    ax1.set_ylabel('Forward Rate (%)')
    ax1.set_title('SABR Model: Single Path - Forward Rate')
    ax1.grid(True, alpha=0.3)
    
    # Volatility path
    ax2.plot(t_points, alpha_path, 'r-')
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Volatility (α)')
    ax2.set_title('SABR Model: Single Path - Volatility')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'sabr_single_path.png'))
    print(f"Saved single path plot to '{os.path.join(PLOTS_DIR, 'sabr_single_path.png')}'")
    
    # 2. Simulate multiple paths
    print("\n2. Simulating multiple paths...")
    n_paths = 50
    
    t_points, forward_paths, alpha_paths = simulate_multiple_sabr_paths(
        sabr_model, t_span, dt, n_paths=n_paths
    )
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    
    # Forward rate paths
    for i in range(n_paths):
        ax1.plot(t_points, forward_paths[i, :] * 100, 'b-', alpha=0.2)  # *100 to show in percentage
    
    # Add mean forward path
    mean_forward_path = np.mean(forward_paths, axis=0) * 100
    ax1.plot(t_points, mean_forward_path, 'r-', linewidth=2, label='Mean Path')
    
    # Add reference to initial forward rate
    ax1.axhline(f0 * 100, color='k', linestyle=':', label=f'Initial rate ({f0:.2%})')
    
    ax1.set_ylabel('Forward Rate (%)')
    ax1.set_title('SABR Model: Multiple Paths - Forward Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volatility paths
    for i in range(n_paths):
        ax2.plot(t_points, alpha_paths[i, :], 'r-', alpha=0.2)
    
    # Add mean volatility path
    mean_alpha_path = np.mean(alpha_paths, axis=0)
    ax2.plot(t_points, mean_alpha_path, 'b-', linewidth=2, label='Mean Volatility')
    
    # Add reference to initial volatility
    ax2.axhline(alpha0, color='k', linestyle=':', label=f'Initial volatility ({alpha0:.2f})')
    
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Volatility (α)')
    ax2.set_title('SABR Model: Multiple Paths - Volatility')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'sabr_multiple_paths.png'))
    print(f"Saved multiple paths plot to '{os.path.join(PLOTS_DIR, 'sabr_multiple_paths.png')}'")
    
    # 3. Demonstrate volatility smile/skew using SABR formula
    print("\n3. Demonstrating volatility smile/skew...")
    
    # Range of strikes
    strikes = np.linspace(0.01, 0.06, 51)  # 1% to 6%
    
    # Different expiry times
    expiries = [0.25, 0.5, 1.0, 2.0]  # 3M, 6M, 1Y, 2Y
    
    # Calculate SABR implied vols for each strike and expiry
    plt.figure(figsize=(12, 6))
    
    for T in expiries:
        implied_vols = np.array([sabr_model.sabr_implied_volatility(K, T) for K in strikes])
        plt.plot(strikes * 100, implied_vols, '-', label=f'T = {T:.1f}Y')
    
    # Add reference lines
    plt.axvline(f0 * 100, color='k', linestyle=':', label=f'ATM rate ({f0:.2%})')
    
    plt.xlabel('Strike Rate (%)')
    plt.ylabel('Implied Volatility')
    plt.title('SABR Model: Volatility Smile/Skew')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, 'sabr_volatility_smile.png'))
    print(f"Saved volatility smile plot to '{os.path.join(PLOTS_DIR, 'sabr_volatility_smile.png')}'")
    
    # 4. Compare different beta parameters
    print("\n4. Comparing different beta parameters...")
    
    # Different beta values
    beta_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    # Fix expiry time
    T = 1.0  # 1 year
    
    # Calculate SABR implied vols for each beta value
    plt.figure(figsize=(12, 6))
    
    for beta_val in beta_values:
        # Create model with this beta
        temp_model = SABRModel(alpha, beta_val, nu, rho, f0, alpha0)
        
        # Calculate implied vols
        implied_vols = np.array([temp_model.sabr_implied_volatility(K, T) for K in strikes])
        plt.plot(strikes * 100, implied_vols, '-', label=f'β = {beta_val:.1f}')
    
    # Add reference line
    plt.axvline(f0 * 100, color='k', linestyle=':', label=f'ATM rate ({f0:.2%})')
    
    plt.xlabel('Strike Rate (%)')
    plt.ylabel('Implied Volatility')
    plt.title('SABR Model: Effect of Beta Parameter on Volatility Smile (T=1Y)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, 'sabr_beta_comparison.png'))
    print(f"Saved beta comparison plot to '{os.path.join(PLOTS_DIR, 'sabr_beta_comparison.png')}'")
    
    # 5. Analyze impact of volatility of volatility (nu)
    print("\n5. Analyzing impact of volatility of volatility...")
    
    # Different nu values
    nu_values = [0.1, 0.2, 0.4, 0.6, 0.8]
    
    # Calculate SABR implied vols for each nu value
    plt.figure(figsize=(12, 6))
    
    for nu_val in nu_values:
        # Create model with this nu
        temp_model = SABRModel(alpha, beta, nu_val, rho, f0, alpha0)
        
        # Calculate implied vols
        implied_vols = np.array([temp_model.sabr_implied_volatility(K, T) for K in strikes])
        plt.plot(strikes * 100, implied_vols, '-', label=f'ν = {nu_val:.1f}')
    
    # Add reference line
    plt.axvline(f0 * 100, color='k', linestyle=':', label=f'ATM rate ({f0:.2%})')
    
    plt.xlabel('Strike Rate (%)')
    plt.ylabel('Implied Volatility')
    plt.title('SABR Model: Effect of Vol-of-Vol Parameter on Volatility Smile (T=1Y)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, 'sabr_nu_comparison.png'))
    print(f"Saved nu comparison plot to '{os.path.join(PLOTS_DIR, 'sabr_nu_comparison.png')}'")
    
    # 6. Analyze impact of correlation (rho)
    print("\n6. Analyzing impact of correlation...")
    
    # Different rho values
    rho_values = [-0.8, -0.5, -0.3, 0.0, 0.3, 0.6]
    
    # Calculate SABR implied vols for each rho value
    plt.figure(figsize=(12, 6))
    
    for rho_val in rho_values:
        # Create model with this rho
        temp_model = SABRModel(alpha, beta, nu, rho_val, f0, alpha0)
        
        # Calculate implied vols
        implied_vols = np.array([temp_model.sabr_implied_volatility(K, T) for K in strikes])
        plt.plot(strikes * 100, implied_vols, '-', label=f'ρ = {rho_val:.1f}')
    
    # Add reference line
    plt.axvline(f0 * 100, color='k', linestyle=':', label=f'ATM rate ({f0:.2%})')
    
    plt.xlabel('Strike Rate (%)')
    plt.ylabel('Implied Volatility')
    plt.title('SABR Model: Effect of Correlation Parameter on Volatility Smile (T=1Y)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, 'sabr_rho_comparison.png'))
    print(f"Saved rho comparison plot to '{os.path.join(PLOTS_DIR, 'sabr_rho_comparison.png')}'")
    
    # 7. Calculate option prices
    print("\n7. Calculating option prices...")
    
    # Define option parameters
    # We'll price caps and floors (interest rate options)
    T = 1.0  # 1 year
    
    # Strikes
    option_strikes = np.linspace(0.01, 0.05, 9)  # 1% to 5%, 9 strikes
    
    # Calculate option prices using SABR implied volatility
    cap_prices = np.zeros(len(option_strikes))
    floor_prices = np.zeros(len(option_strikes))
    
    for i, K in enumerate(option_strikes):
        # Get SABR implied volatility for this strike
        implied_vol = sabr_model.sabr_implied_volatility(K, T)
        
        # Calculate cap price (call option on forward rate)
        cap_prices[i] = calculate_black_price(f0, K, T, implied_vol, is_call=True)
        
        # Calculate floor price (put option on forward rate)
        floor_prices[i] = calculate_black_price(f0, K, T, implied_vol, is_call=False)
    
    # Calculate prices using constant volatility (Black model)
    black_cap_prices = np.zeros(len(option_strikes))
    black_floor_prices = np.zeros(len(option_strikes))
    
    for i, K in enumerate(option_strikes):
        # Use constant volatility = alpha0
        black_cap_prices[i] = calculate_black_price(f0, K, T, alpha0, is_call=True)
        black_floor_prices[i] = calculate_black_price(f0, K, T, alpha0, is_call=False)
    
    # Plot option prices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cap prices (calls)
    ax1.plot(option_strikes * 100, cap_prices * 100, 'bo-', label='SABR Model')
    ax1.plot(option_strikes * 100, black_cap_prices * 100, 'ro--', label='Black Model')
    ax1.set_xlabel('Strike Rate (%)')
    ax1.set_ylabel('Option Price (% of Notional)')
    ax1.set_title('Cap Prices (Call Options on Forward Rate)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Floor prices (puts)
    ax2.plot(option_strikes * 100, floor_prices * 100, 'bo-', label='SABR Model')
    ax2.plot(option_strikes * 100, black_floor_prices * 100, 'ro--', label='Black Model')
    ax2.set_xlabel('Strike Rate (%)')
    ax2.set_ylabel('Option Price (% of Notional)')
    ax2.set_title('Floor Prices (Put Options on Forward Rate)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'sabr_option_prices.png'))
    print(f"Saved option prices plot to '{os.path.join(PLOTS_DIR, 'sabr_option_prices.png')}'")
    
    # Print some option prices
    print(f"\nOption Prices (1-year maturity, F₀={f0:.2%}):")
    print(f"{'Strike':^10}|{'SABR Cap':^15}|{'Black Cap':^15}|{'SABR Floor':^15}|{'Black Floor':^15}")
    print("-" * 72)
    
    for i, K in enumerate(option_strikes):
        print(f"{K*100:^10.2f}%|{cap_prices[i]*100:^15.4f}%|{black_cap_prices[i]*100:^15.4f}%|{floor_prices[i]*100:^15.4f}%|{black_floor_prices[i]*100:^15.4f}%")
    
    print("\nSimulation complete! All plots saved to the current directory.") 
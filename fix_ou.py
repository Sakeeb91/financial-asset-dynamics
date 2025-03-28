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

# Create output directories
PLOTS_DIR = "ou_simulation/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Simple implementation of the Ornstein-Uhlenbeck model
class OrnsteinUhlenbeck:
    def __init__(self, theta, mu, sigma):
        """
        Initialize the Ornstein-Uhlenbeck model.
        
        Args:
            theta: Speed of mean reversion
            mu: Long-term mean
            sigma: Volatility
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
    
    def drift(self, x, t):
        """Drift coefficient function."""
        return self.theta * (self.mu - x)
    
    def diffusion(self, x, t):
        """Diffusion coefficient function."""
        return self.sigma
    
    def exact_solution(self, t, W, x0):
        """
        Calculate the exact solution of the OU process.
        
        Args:
            t: Time points
            W: Wiener process increments
            x0: Initial value
            
        Returns:
            Exact solution path
        """
        # Ensure t starts with 0
        if t[0] != 0:
            t = np.concatenate([[0], t])
            W = np.concatenate([[0], W])
        
        # Calculate integrated Wiener process
        integral = np.zeros_like(t)
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            dW = W[i] - W[i-1]
            integral[i] = integral[i-1] * np.exp(-self.theta * dt) + self.sigma * dW
            
        # Calculate exact solution
        x = x0 * np.exp(-self.theta * t) + self.mu * (1 - np.exp(-self.theta * t)) + integral
        return x
    
    def stationary_distribution_parameters(self):
        """Return mean and variance of the stationary distribution."""
        return self.mu, self.sigma**2 / (2 * self.theta)

class SimulationEngine:
    def __init__(self, model, x0, t_span, dt):
        """
        Initialize simulation engine.
        
        Args:
            model: SDE model
            x0: Initial value
            t_span: (t_start, t_end) time interval
            dt: Time step size
        """
        self.model = model
        self.x0 = x0
        self.t_span = t_span
        self.dt = dt
        
        # Precompute time points
        t_start, t_end = t_span
        self.n_steps = int((t_end - t_start) / dt) + 1
        self.t_points = np.linspace(t_start, t_end, self.n_steps)
    
    def run_simulation(self, method='euler', n_paths=1, random_state=None):
        """
        Run simulation using specified method.
        
        Args:
            method: Numerical method ('euler' or 'milstein')
            n_paths: Number of paths to simulate
            random_state: Random seed
            
        Returns:
            tuple: (t_points, x_paths) time points and simulated paths
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        # Initialize paths
        x_paths = np.zeros((n_paths, self.n_steps))
        x_paths[:, 0] = self.x0
        
        # Simulate paths
        for path_idx in range(n_paths):
            for i in range(1, self.n_steps):
                t = self.t_points[i-1]
                x = x_paths[path_idx, i-1]
                dt = self.dt
                dW = np.random.normal(0, np.sqrt(dt))
                
                if method == 'euler':
                    x_paths[path_idx, i] = self._euler_step(x, t, dt, dW)
                elif method == 'milstein':
                    x_paths[path_idx, i] = self._milstein_step(x, t, dt, dW)
                else:
                    raise ValueError(f"Unknown method: {method}")
        
        # Return time points and simulated paths
        if n_paths == 1:
            return self.t_points, x_paths[0]
        else:
            return self.t_points, x_paths
    
    def _euler_step(self, x, t, dt, dW):
        """Euler-Maruyama step."""
        drift = self.model.drift(x, t) * dt
        diffusion = self.model.diffusion(x, t) * dW
        return x + drift + diffusion
    
    def _milstein_step(self, x, t, dt, dW):
        """Milstein step."""
        drift = self.model.drift(x, t) * dt
        diffusion = self.model.diffusion(x, t)
        diffusion_term = diffusion * dW
        # For OU, the diffusion derivative is 0, so Milstein is the same as Euler
        return x + drift + diffusion_term
    
    def run_exact_solution(self, n_paths=1, random_state=None):
        """
        Compute exact solution for comparison.
        
        Args:
            n_paths: Number of paths to simulate
            random_state: Random seed
            
        Returns:
            tuple: (t_points, x_paths) time points and exact paths
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        # Initialize paths
        x_paths = np.zeros((n_paths, self.n_steps))
        
        # Generate Wiener processes
        W = np.zeros((n_paths, self.n_steps))
        for i in range(1, self.n_steps):
            W[:, i] = W[:, i-1] + np.random.normal(0, np.sqrt(self.dt), size=n_paths)
        
        # Compute exact solutions
        for path_idx in range(n_paths):
            x_paths[path_idx] = self.model.exact_solution(self.t_points, W[path_idx], self.x0)
        
        # Return time points and exact paths
        if n_paths == 1:
            return self.t_points, x_paths[0]
        else:
            return self.t_points, x_paths
    
    def compare_methods(self, n_paths=1, random_state=None):
        """
        Compare different numerical methods.
        
        Args:
            n_paths: Number of paths for each method
            random_state: Random seed
            
        Returns:
            dict: {'method': (t_points, x_path)} for each method
        """
        results = {}
        
        # Run simulations with different methods
        for method in ['euler', 'milstein']:
            results[method] = self.run_simulation(method=method, n_paths=n_paths, random_state=random_state)
        
        # Add exact solution
        results['exact'] = self.run_exact_solution(n_paths=n_paths, random_state=random_state)
        
        return results
    
    def compare_error(self, dt_values, final_time_only=False, n_paths=100, random_state=None):
        """
        Compare errors for different time step sizes.
        
        Args:
            dt_values: Array of time step sizes to test
            final_time_only: If True, only compare at final time point
            n_paths: Number of paths to use for error estimation
            random_state: Random seed
            
        Returns:
            dict: {'method': [error_stats]} for each method and dt
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        errors = {'euler': [], 'milstein': []}
        
        for dt in dt_values:
            # Create a new engine with the current dt
            engine = SimulationEngine(self.model, self.x0, self.t_span, dt)
            
            # Generate paths using each method
            for method in ['euler', 'milstein']:
                method_errors = []
                
                for _ in range(n_paths):
                    # Generate a random seed for this path
                    path_seed = np.random.randint(0, 10000)
                    
                    # Compute numerical and exact solutions
                    _, x_numerical = engine.run_simulation(method=method, random_state=path_seed)
                    _, x_exact = engine.run_exact_solution(random_state=path_seed)
                    
                    # Calculate error
                    if final_time_only:
                        error = abs(x_numerical[-1] - x_exact[-1])
                    else:
                        error = np.mean(abs(x_numerical - x_exact))
                    
                    method_errors.append(error)
                
                # Calculate error statistics
                mean_error = np.mean(method_errors)
                std_error = np.std(method_errors)
                
                errors[method].append({'mean': mean_error, 'std': std_error})
        
        return errors


# Main simulation function
if __name__ == "__main__":
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
    print(f"Saved single path plot to '{PLOTS_DIR}/ou_single_path.png'")

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
    print(f"Saved multiple paths plot to '{PLOTS_DIR}/ou_multiple_paths.png'")

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
    print(f"Saved distribution plots to '{PLOTS_DIR}/ou_distributions.png'")

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
    print(f"Saved methods comparison plot to '{PLOTS_DIR}/ou_methods_comparison.png'")

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
    print(f"Saved mean-reversion demonstration plot to '{PLOTS_DIR}/ou_mean_reversion.png'")

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
    print(f"Saved convergence analysis plot to '{PLOTS_DIR}/ou_convergence.png'")

    print(f"\nSimulation complete! All plots saved to the {PLOTS_DIR} directory.") 
#!/usr/bin/env python
"""
Ornstein-Uhlenbeck Process Dashboard Creator
This script combines all OU simulation plots into a single dashboard.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import numpy as np
import matplotlib.gridspec as gridspec

def check_plots():
    """Check if all required plots exist, if not, run the simulation."""
    plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots', 'ou')
    required_plots = [
        'ou_single_path.png',
        'ou_multiple_paths.png',
        'ou_distributions.png',
        'ou_methods_comparison.png',
        'ou_mean_reversion.png',
        'ou_convergence.png'
    ]
    
    # Check if any plots are missing
    missing_plots = [plot for plot in required_plots if not os.path.exists(os.path.join(plot_dir, plot))]
    
    if missing_plots:
        print(f"Missing plots: {missing_plots}")
        print("Running OU simulation to generate plots...")
        simulation_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'simulations', 'ou_simulation.py')
        os.system(f'python {simulation_script}')
        
        # Verify plots were generated
        still_missing = [plot for plot in required_plots if not os.path.exists(os.path.join(plot_dir, plot))]
        if still_missing:
            print(f"Error: Could not generate plots: {still_missing}")
            return False
    else:
        print("All required plots found.")
    
    return True

def create_dashboard():
    """Create a dashboard combining all OU simulation plots."""
    plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots', 'ou')
    
    # Create figure
    fig = plt.figure(figsize=(16, 14))
    
    # Create grid for subplots
    grid = gridspec.GridSpec(3, 2, figure=fig)
    
    # Add title to the figure
    fig.suptitle('Ornstein-Uhlenbeck Process Simulation Dashboard', fontsize=20, y=0.98)
    
    # Plot names and titles
    plots = [
        {'file': 'ou_single_path.png', 'title': 'Single Path Simulation', 'pos': (0, 0)},
        {'file': 'ou_multiple_paths.png', 'title': 'Multiple Paths Simulation', 'pos': (0, 1)},
        {'file': 'ou_distributions.png', 'title': 'Distribution Analysis', 'pos': (1, 0)},
        {'file': 'ou_methods_comparison.png', 'title': 'Numerical Methods Comparison', 'pos': (1, 1)},
        {'file': 'ou_mean_reversion.png', 'title': 'Mean Reversion Demonstration', 'pos': (2, 0)},
        {'file': 'ou_convergence.png', 'title': 'Convergence Analysis', 'pos': (2, 1)}
    ]
    
    # Load and place images
    for plot in plots:
        i, j = plot['pos']
        ax = fig.add_subplot(grid[i, j])
        img = mpimg.imread(os.path.join(plot_dir, plot['file']))
        ax.imshow(img)
        ax.set_title(plot['title'])
        ax.axis('off')
    
    # Add text with summary information
    summary_text = """
    Ornstein-Uhlenbeck Process Summary:
    - Parameters: θ=5.0 (speed of mean reversion), μ=0.07 (long-term mean), σ=0.02 (volatility), X₀=0.10
    - Mean-reverting stochastic process that tends toward its long-term mean value
    - Half-life of deviations is approximately 0.14 years (ln(2)/θ)
    - Process distribution converges to normal distribution with mean μ and standard deviation σ/√(2θ)
    - Euler-Maruyama and Milstein methods show similar performance for this process
    - Mean reversion is demonstrated by paths starting from different initial values
    """
    fig.text(0.5, 0.01, summary_text, ha='center', va='bottom', fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save the dashboard
    output_path = os.path.join(plot_dir, 'ou_simulation_dashboard.png')
    plt.savefig(output_path)
    print(f"Dashboard created and saved as '{output_path}'")
    plt.close()

if __name__ == "__main__":
    if check_plots():
        create_dashboard()
    else:
        print("Could not create dashboard due to missing plots.") 
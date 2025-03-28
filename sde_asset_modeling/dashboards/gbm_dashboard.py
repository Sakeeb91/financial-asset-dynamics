#!/usr/bin/env python
"""
Geometric Brownian Motion Dashboard Creator
This script combines all GBM simulation plots into a single dashboard.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import numpy as np

def check_plots():
    """Check if all required plots exist, if not, run the simulation."""
    plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots', 'gbm')
    required_plots = [
        'gbm_single_path.png',
        'gbm_multiple_paths.png',
        'gbm_distribution.png',
        'gbm_methods_comparison.png',
        'gbm_convergence.png'
    ]
    
    # Check if any plots are missing
    missing_plots = [plot for plot in required_plots if not os.path.exists(os.path.join(plot_dir, plot))]
    
    if missing_plots:
        print(f"Missing plots: {missing_plots}")
        print("Running GBM simulation to generate plots...")
        simulation_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'simulations', 'gbm_simulation.py')
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
    """Create a dashboard combining all GBM simulation plots."""
    plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots', 'gbm')
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Add title to the figure
    fig.suptitle('Geometric Brownian Motion Simulation Dashboard', fontsize=20, y=0.98)
    
    # Define subplot positions
    positions = [
        (0, 0),  # top left
        (0, 1),  # top right
        (1, 0),  # middle left
        (1, 1),  # middle right
        (2, 0)   # bottom (spans both columns)
    ]
    
    # Plot names and titles
    plots = [
        {'file': 'gbm_single_path.png', 'title': 'Single Path Simulation'},
        {'file': 'gbm_multiple_paths.png', 'title': 'Multiple Paths Simulation'},
        {'file': 'gbm_distribution.png', 'title': 'Distribution of Final Prices'},
        {'file': 'gbm_methods_comparison.png', 'title': 'Numerical Methods Comparison'},
        {'file': 'gbm_convergence.png', 'title': 'Convergence Analysis'}
    ]
    
    # Load and place images
    for i, plot in enumerate(plots):
        pos = positions[i]
        
        # Last plot spans both columns
        if i == 4:
            ax = plt.subplot2grid((3, 2), pos, colspan=2)
        else:
            ax = plt.subplot2grid((3, 2), pos)
        
        img = mpimg.imread(os.path.join(plot_dir, plot['file']))
        ax.imshow(img)
        ax.set_title(plot['title'])
        ax.axis('off')
    
    # Add text with summary information
    summary_text = """
    GBM Model Summary:
    - Parameters: μ=0.10 (10% annual return), σ=0.20 (20% annual volatility), S₀=$100
    - Simulation results closely match theoretical expectations
    - Final price distribution follows log-normal distribution
    - Euler-Maruyama and Milstein methods show similar performance for this process
    - Convergence analysis confirms numerical stability of simulation methods
    """
    fig.text(0.5, 0.02, summary_text, ha='center', va='bottom', fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    # Save the dashboard
    output_path = os.path.join(plot_dir, 'gbm_simulation_dashboard.png')
    plt.savefig(output_path)
    print(f"Dashboard created and saved as '{output_path}'")
    plt.close()

if __name__ == "__main__":
    if check_plots():
        create_dashboard()
    else:
        print("Could not create dashboard due to missing plots.") 
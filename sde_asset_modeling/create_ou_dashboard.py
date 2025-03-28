#!/usr/bin/env python
"""
Dashboard Generator for Ornstein-Uhlenbeck (OU) Process Simulation Results
This script creates a dashboard by combining all OU simulation plots into a single image.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from matplotlib.gridspec import GridSpec

# Define directories
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots', 'ou')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Function to check if plots exist or need to be regenerated
def check_plots():
    """
    Check if all required OU simulation plots exist.
    
    Returns:
        bool: True if all plots exist, False otherwise
    """
    required_plots = [
        'ou_single_path.png', 
        'ou_multiple_paths.png', 
        'ou_distributions.png',
        'ou_methods_comparison.png', 
        'ou_mean_reversion.png',
        'ou_convergence.png'
    ]
    
    # Check if each plot exists in the plots directory
    all_exist = True
    missing_plots = []
    
    for plot in required_plots:
        plot_path = os.path.join(PLOTS_DIR, plot)
        if not os.path.exists(plot_path):
            all_exist = False
            missing_plots.append(plot)
    
    if not all_exist:
        print("The following plots are missing:")
        for plot in missing_plots:
            print(f"  - {plot}")
        print("Running simulation to generate plots...")
        os.system("python run_ou_simulation.py")
        
        # Verify plots were generated
        still_missing = []
        for plot in required_plots:
            plot_path = os.path.join(PLOTS_DIR, plot)
            if not os.path.exists(plot_path):
                still_missing.append(plot)
                
        if still_missing:
            print(f"Error: Could not generate the following plots: {still_missing}")
            return False
    else:
        print("All required plots found.")
        
    return True

# Create dashboard
def create_dashboard():
    """Create a dashboard combining all OU simulation plots."""
    # Check if all plots exist
    if not check_plots():
        print("Dashboard creation aborted due to missing plots.")
        return
    
    print("Creating OU simulation dashboard...")
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1.2, 1])
    
    # Title for the dashboard
    fig.suptitle('Ornstein-Uhlenbeck (OU) Process Simulation Dashboard', fontsize=20)
    
    # Add plots to the dashboard
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = mpimg.imread(os.path.join(PLOTS_DIR, 'ou_single_path.png'))
    ax1.imshow(img1)
    ax1.axis('off')
    ax1.set_title('Single Path Simulation', fontsize=14)
    
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = mpimg.imread(os.path.join(PLOTS_DIR, 'ou_multiple_paths.png'))
    ax2.imshow(img2)
    ax2.axis('off')
    ax2.set_title('Multiple Paths Simulation', fontsize=14)
    
    ax3 = fig.add_subplot(gs[1, 0:2])
    img3 = mpimg.imread(os.path.join(PLOTS_DIR, 'ou_distributions.png'))
    ax3.imshow(img3)
    ax3.axis('off')
    ax3.set_title('Distribution at Different Time Points', fontsize=14)
    
    ax4 = fig.add_subplot(gs[2, 0])
    img4 = mpimg.imread(os.path.join(PLOTS_DIR, 'ou_mean_reversion.png'))
    ax4.imshow(img4)
    ax4.axis('off')
    ax4.set_title('Mean-Reversion Demonstration', fontsize=14)
    
    ax5 = fig.add_subplot(gs[2, 1])
    img5 = mpimg.imread(os.path.join(PLOTS_DIR, 'ou_methods_comparison.png'))
    ax5.imshow(img5)
    ax5.axis('off')
    ax5.set_title('Numerical Methods Comparison', fontsize=14)
    
    # Add summary text
    plt.figtext(0.5, 0.01, 
                'OU Process Parameters: θ=5.0 (speed of mean reversion), μ=0.07 (7% long-term mean), σ=0.02 (volatility)\n'
                'Half-life of deviations: 0.1386 years. The mean-reverting nature is clearly visible in all simulations.\n'
                'The process converges to a stationary normal distribution with mean μ and variance σ²/(2θ).', 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave room for title and summary
    dashboard_path = os.path.join(PLOTS_DIR, 'ou_dashboard.png')
    plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
    print(f"Dashboard created and saved to '{dashboard_path}'")

if __name__ == "__main__":
    create_dashboard() 
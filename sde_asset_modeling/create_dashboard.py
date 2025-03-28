#!/usr/bin/env python
"""
Dashboard Generator for GBM Simulation Results
This script creates a dashboard by combining all simulation plots into a single image.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from matplotlib.gridspec import GridSpec

# Function to check if plots exist or need to be regenerated
def check_plots():
    required_plots = [
        'gbm_single_path.png', 
        'gbm_multiple_paths.png', 
        'gbm_distribution.png',
        'gbm_methods_comparison.png', 
        'gbm_convergence.png'
    ]
    
    missing = [plot for plot in required_plots if not os.path.exists(plot)]
    if missing:
        print(f"Missing plots: {missing}")
        print("Running simulation to generate plots...")
        os.system("python run_gbm_simulation.py")
    else:
        print("All plots found.")

# Create dashboard
def create_dashboard():
    # Check if all plots exist
    check_plots()
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])
    
    # Title for the dashboard
    fig.suptitle('Geometric Brownian Motion (GBM) Simulation Dashboard', fontsize=20)
    
    # Add plots to the dashboard
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = mpimg.imread('gbm_single_path.png')
    ax1.imshow(img1)
    ax1.axis('off')
    ax1.set_title('Single Path Simulation', fontsize=14)
    
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = mpimg.imread('gbm_multiple_paths.png')
    ax2.imshow(img2)
    ax2.axis('off')
    ax2.set_title('Multiple Paths Simulation', fontsize=14)
    
    ax3 = fig.add_subplot(gs[1, 0:2])
    img3 = mpimg.imread('gbm_distribution.png')
    ax3.imshow(img3)
    ax3.axis('off')
    ax3.set_title('Distribution of Final Prices', fontsize=14)
    
    ax4 = fig.add_subplot(gs[2, 0])
    img4 = mpimg.imread('gbm_methods_comparison.png')
    ax4.imshow(img4)
    ax4.axis('off')
    ax4.set_title('Numerical Methods Comparison', fontsize=14)
    
    ax5 = fig.add_subplot(gs[2, 1])
    img5 = mpimg.imread('gbm_convergence.png')
    ax5.imshow(img5)
    ax5.axis('off')
    ax5.set_title('Convergence Analysis', fontsize=14)
    
    # Add summary text
    plt.figtext(0.5, 0.01, 
                'GBM Model Parameters: μ=0.10 (10% annual return), σ=0.20 (20% annual volatility), S₀=$100\n'
                'Simulation results show close agreement between theoretical and empirical distributions.\n'
                'Euler-Maruyama and Milstein methods show similar performance for this stochastic process.', 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave room for title and summary
    plt.savefig('gbm_simulation_dashboard.png', dpi=150, bbox_inches='tight')
    print("Dashboard created and saved as 'gbm_simulation_dashboard.png'")

if __name__ == "__main__":
    create_dashboard() 
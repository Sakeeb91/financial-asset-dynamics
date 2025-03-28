#!/usr/bin/env python
"""
SABR Model Dashboard Creation Script
This script creates a dashboard of SABR model simulation results.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import numpy as np

def check_plots():
    """
    Check if all required plots exist.
    
    Returns:
        bool: True if all plots exist, False otherwise
    """
    # Set the plot directory
    plot_dir = os.path.join(os.path.dirname(__file__), 'plots', 'sabr')
    
    # List of required plots
    required_plots = [
        'sabr_single_path.png',
        'sabr_multiple_paths.png',
        'sabr_volatility_smile.png',
        'sabr_beta_comparison.png',
        'sabr_nu_comparison.png',
        'sabr_rho_comparison.png',
        'sabr_option_prices.png'
    ]
    
    # Check if all required plots exist
    all_exist = True
    missing_plots = []
    
    for plot in required_plots:
        plot_path = os.path.join(plot_dir, plot)
        if not os.path.exists(plot_path):
            all_exist = False
            missing_plots.append(plot)
    
    if not all_exist:
        print("The following plots are missing:")
        for plot in missing_plots:
            print(f"  - {plot}")
        print("Please run the SABR simulation script first.")
    
    return all_exist

def create_dashboard():
    """
    Create a dashboard combining all SABR model simulation plots.
    """
    # Set the plot directory
    plot_dir = os.path.join(os.path.dirname(__file__), 'plots', 'sabr')
    
    # Create a figure with a grid layout
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig)
    
    # Title
    fig.suptitle('SABR Model Simulation Dashboard', fontsize=24)
    
    # Add a summary of model parameters
    params_text = """
    SABR Model Parameters:
    - Initial Forward Rate (F₀): 3.00%
    - Initial Volatility (α₀): 0.15
    - CEV Parameter (β): 0.50
    - Volatility of Volatility (ν): 0.40
    - Correlation (ρ): -0.30
    
    The SABR model captures the volatility smile/skew observed in interest rate options markets.
    β controls the relationship between volatility and rate level.
    ν controls how much the volatility itself varies.
    ρ controls the correlation between rate and volatility processes.
    """
    
    # Add the parameter summary to the bottom right
    ax_params = fig.add_subplot(gs[2, 2])
    ax_params.text(0.5, 0.5, params_text, ha='center', va='center', fontsize=12)
    ax_params.axis('off')
    
    # Load and place each plot image
    # Single path
    ax1 = fig.add_subplot(gs[0, 0])
    img = mpimg.imread(os.path.join(plot_dir, 'sabr_single_path.png'))
    ax1.imshow(img)
    ax1.set_title('Single Path Simulation', fontsize=14)
    ax1.axis('off')
    
    # Multiple paths
    ax2 = fig.add_subplot(gs[0, 1:])
    img = mpimg.imread(os.path.join(plot_dir, 'sabr_multiple_paths.png'))
    ax2.imshow(img)
    ax2.set_title('Multiple Paths Simulation', fontsize=14)
    ax2.axis('off')
    
    # Volatility smile
    ax3 = fig.add_subplot(gs[1, 0])
    img = mpimg.imread(os.path.join(plot_dir, 'sabr_volatility_smile.png'))
    ax3.imshow(img)
    ax3.set_title('Volatility Smile/Skew', fontsize=14)
    ax3.axis('off')
    
    # Beta comparison
    ax4 = fig.add_subplot(gs[1, 1])
    img = mpimg.imread(os.path.join(plot_dir, 'sabr_beta_comparison.png'))
    ax4.imshow(img)
    ax4.set_title('Beta Parameter Effect', fontsize=14)
    ax4.axis('off')
    
    # Nu comparison
    ax5 = fig.add_subplot(gs[1, 2])
    img = mpimg.imread(os.path.join(plot_dir, 'sabr_nu_comparison.png'))
    ax5.imshow(img)
    ax5.set_title('Volatility of Volatility Effect', fontsize=14)
    ax5.axis('off')
    
    # Rho comparison
    ax6 = fig.add_subplot(gs[2, 0])
    img = mpimg.imread(os.path.join(plot_dir, 'sabr_rho_comparison.png'))
    ax6.imshow(img)
    ax6.set_title('Correlation Effect', fontsize=14)
    ax6.axis('off')
    
    # Option prices
    ax7 = fig.add_subplot(gs[2, 1])
    img = mpimg.imread(os.path.join(plot_dir, 'sabr_option_prices.png'))
    ax7.imshow(img)
    ax7.set_title('Option Pricing', fontsize=14)
    ax7.axis('off')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the title
    
    # Save the dashboard
    plt.savefig(os.path.join(plot_dir, 'sabr_dashboard.png'), dpi=150)
    print(f"Dashboard created and saved to '{os.path.join(plot_dir, 'sabr_dashboard.png')}'")
    plt.close()

if __name__ == "__main__":
    print("Checking for required SABR simulation plots...")
    if check_plots():
        print("All required plots found. Creating dashboard...")
        create_dashboard()
        print("Dashboard creation complete!")
    else:
        print("Dashboard creation aborted due to missing plots.") 
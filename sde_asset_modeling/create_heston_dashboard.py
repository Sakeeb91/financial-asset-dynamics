#!/usr/bin/env python
"""
Heston Stochastic Volatility Dashboard Creator
This script combines all Heston model simulation plots into a single dashboard.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import numpy as np
import matplotlib.gridspec as gridspec

def check_plots():
    """Check if all required plots exist, if not, run the simulation."""
    required_plots = [
        'heston_single_path.png',
        'heston_multiple_paths.png',
        'heston_distribution.png',
        'heston_parameter_comparison.png',
        'heston_volatility_clustering.png',
        'heston_option_pricing.png'
    ]
    
    # Check if any plots are missing
    missing_plots = [plot for plot in required_plots if not os.path.exists(plot)]
    
    if missing_plots:
        print(f"Missing plots: {missing_plots}")
        print("Running Heston model simulation to generate plots...")
        os.system('python run_heston_simulation.py')
        
        # Verify plots were generated
        still_missing = [plot for plot in required_plots if not os.path.exists(plot)]
        if still_missing:
            print(f"Error: Could not generate plots: {still_missing}")
            return False
    else:
        print("All required plots found.")
    
    return True

def create_dashboard():
    """Create a dashboard combining all Heston model simulation plots."""
    # Create figure
    fig = plt.figure(figsize=(18, 16))
    
    # Create grid for subplots
    grid = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.2])
    
    # Add title to the figure
    fig.suptitle('Heston Stochastic Volatility Model Simulation Dashboard', fontsize=20, y=0.98)
    
    # Load and place images
    # 1. Single Path (top left)
    ax1 = fig.add_subplot(grid[0, 0])
    img1 = mpimg.imread('heston_single_path.png')
    ax1.imshow(img1)
    ax1.set_title('Single Path Simulation')
    ax1.axis('off')
    
    # 2. Multiple Paths (top right)
    ax2 = fig.add_subplot(grid[0, 1])
    img2 = mpimg.imread('heston_multiple_paths.png')
    ax2.imshow(img2)
    ax2.set_title('Multiple Paths Simulation')
    ax2.axis('off')
    
    # 3. Distribution of final prices (middle left)
    ax3 = fig.add_subplot(grid[1, 0])
    img3 = mpimg.imread('heston_distribution.png')
    ax3.imshow(img3)
    ax3.set_title('Distribution of Final Prices')
    ax3.axis('off')
    
    # 4. Parameter comparison (middle right)
    ax4 = fig.add_subplot(grid[1, 1])
    img4 = mpimg.imread('heston_parameter_comparison.png')
    ax4.imshow(img4)
    ax4.set_title('Parameter Comparison')
    ax4.axis('off')
    
    # 5. Volatility clustering (bottom left)
    ax5 = fig.add_subplot(grid[2, 0])
    img5 = mpimg.imread('heston_volatility_clustering.png')
    ax5.imshow(img5)
    ax5.set_title('Volatility Clustering')
    ax5.axis('off')
    
    # 6. Option pricing (bottom right)
    ax6 = fig.add_subplot(grid[2, 1])
    img6 = mpimg.imread('heston_option_pricing.png')
    ax6.imshow(img6)
    ax6.set_title('Option Pricing & Volatility Smile')
    ax6.axis('off')
    
    # Add text with summary information
    summary_text = """
    Heston Stochastic Volatility Model Summary:
    - Base parameters: μ=0.10 (10% annual return), S₀=$100, volatility characteristics: κ=2.0, θ=0.04, ξ=0.3, ρ=-0.7
    - The model features stochastic volatility that mean-reverts to a long-term level (20% in these simulations)
    - Negative correlation between price and volatility (ρ=-0.7) creates the "leverage effect" seen in equity markets
    - Volatility clustering is demonstrated through periods of high/low volatility that persist over time
    - Stochastic volatility creates fat-tailed return distributions compared to standard GBM
    - The model produces volatility skew/smile patterns in option prices, unlike the flat implied volatility of Black-Scholes
    """
    fig.text(0.5, 0.02, summary_text, ha='center', va='bottom', fontsize=10)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    # Save the dashboard
    plt.savefig('heston_dashboard.png', dpi=150)
    print("Dashboard created and saved as 'heston_dashboard.png'")
    plt.close()

if __name__ == "__main__":
    if check_plots():
        create_dashboard()
    else:
        print("Could not create dashboard due to missing plots.") 
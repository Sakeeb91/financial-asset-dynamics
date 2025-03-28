#!/usr/bin/env python
"""
Jump Diffusion Dashboard Creator
This script combines all Jump Diffusion model simulation plots into a single dashboard.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import numpy as np
import matplotlib.gridspec as gridspec

def check_plots():
    """Check if all required plots exist, if not, run the simulation."""
    plot_dir = os.path.join(os.path.dirname(__file__), 'plots', 'jump_diffusion')
    required_plots = [
        'jump_diffusion_single_path.png',
        'jump_diffusion_multiple_paths.png',
        'jump_diffusion_distribution.png',
        'jump_diffusion_comparison.png',
        'jump_diffusion_option_prices.png'
    ]
    
    # Check if any plots are missing
    missing_plots = [plot for plot in required_plots if not os.path.exists(os.path.join(plot_dir, plot))]
    
    if missing_plots:
        print(f"Missing plots: {missing_plots}")
        print("Running Jump Diffusion simulation to generate plots...")
        os.system('python jump_diffusion_simulation.py')
        
        # Verify plots were generated
        still_missing = [plot for plot in required_plots if not os.path.exists(os.path.join(plot_dir, plot))]
        if still_missing:
            print(f"Error: Could not generate plots: {still_missing}")
            return False
    else:
        print("All required plots found.")
    
    return True

def create_dashboard():
    """Create a dashboard combining all Jump Diffusion simulation plots."""
    # Define plot directory
    plot_dir = os.path.join(os.path.dirname(__file__), 'plots', 'jump_diffusion')
    
    # Create figure
    fig = plt.figure(figsize=(18, 14))
    
    # Create grid for subplots
    grid = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.2])
    
    # Add title to the figure
    fig.suptitle('Jump Diffusion (Merton) Model Simulation Dashboard', fontsize=20, y=0.98)
    
    # Load and place images
    # 1. Single Path (top left)
    ax1 = fig.add_subplot(grid[0, 0])
    img1 = mpimg.imread(os.path.join(plot_dir, 'jump_diffusion_single_path.png'))
    ax1.imshow(img1)
    ax1.set_title('Single Path Simulation with Jumps')
    ax1.axis('off')
    
    # 2. Multiple Paths (top right)
    ax2 = fig.add_subplot(grid[0, 1])
    img2 = mpimg.imread(os.path.join(plot_dir, 'jump_diffusion_multiple_paths.png'))
    ax2.imshow(img2)
    ax2.set_title('Multiple Paths Simulation')
    ax2.axis('off')
    
    # 3. Distribution (middle left)
    ax3 = fig.add_subplot(grid[1, 0])
    img3 = mpimg.imread(os.path.join(plot_dir, 'jump_diffusion_distribution.png'))
    ax3.imshow(img3)
    ax3.set_title('Distribution of Final Prices')
    ax3.axis('off')
    
    # 4. Jump Parameter Comparison (middle right)
    ax4 = fig.add_subplot(grid[1, 1])
    img4 = mpimg.imread(os.path.join(plot_dir, 'jump_diffusion_comparison.png'))
    ax4.imshow(img4)
    ax4.set_title('Comparison of Jump Parameters')
    ax4.axis('off')
    
    # 5. Option Prices (bottom, spanning both columns)
    ax5 = fig.add_subplot(grid[2, :])
    img5 = mpimg.imread(os.path.join(plot_dir, 'jump_diffusion_option_prices.png'))
    ax5.imshow(img5)
    ax5.set_title('Option Pricing: Jump Diffusion vs. Black-Scholes')
    ax5.axis('off')
    
    # Add text with summary information
    summary_text = """
    Merton Jump Diffusion Model Summary:
    - Base model: μ=0.10 (10% annual return), σ=0.20 (20% annual volatility), S₀=$100
    - Jump parameters: λ=5.0 (5 jumps per year), mean jump size=-4.88%, jump size std=0.10
    - The model adds sudden jumps to the standard GBM process, creating fat tails in the return distribution
    - Jump diffusion captures market reactions to sudden events (crashes, earnings surprises, etc.)
    - Option pricing shows impact of jumps on market prices, especially for out-of-the-money options
    - Black-Scholes model typically underprices options when jump risk is present in the market
    """
    fig.text(0.5, 0.02, summary_text, ha='center', va='bottom', fontsize=10)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    # Save the dashboard
    plt.savefig(os.path.join(plot_dir, 'jump_diffusion_dashboard.png'), dpi=150)
    print(f"Dashboard created and saved as '{os.path.join(plot_dir, 'jump_diffusion_dashboard.png')}'")
    plt.close()

if __name__ == "__main__":
    if check_plots():
        create_dashboard()
    else:
        print("Could not create dashboard due to missing plots.") 
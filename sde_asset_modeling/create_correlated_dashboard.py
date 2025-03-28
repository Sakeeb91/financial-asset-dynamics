#!/usr/bin/env python
"""
Correlated Assets Dashboard Creator
This script combines all correlated asset simulation plots into a single dashboard.
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
        'correlated_single_path.png',
        'correlated_multiple_paths.png',
        'correlated_correlations.png',
        'correlated_portfolio.png',
        'correlated_portfolio_distribution.png'
    ]
    
    # Check if any plots are missing
    missing_plots = [plot for plot in required_plots if not os.path.exists(plot)]
    
    if missing_plots:
        print(f"Missing plots: {missing_plots}")
        print("Running correlated simulation to generate plots...")
        os.system('python run_correlated_simulation.py')
        
        # Verify plots were generated
        still_missing = [plot for plot in required_plots if not os.path.exists(plot)]
        if still_missing:
            print(f"Error: Could not generate plots: {still_missing}")
            return False
    else:
        print("All required plots found.")
    
    return True

def create_dashboard():
    """Create a dashboard combining all correlated asset simulation plots."""
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid for subplots
    grid = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1.2, 1])
    
    # Add title to the figure
    fig.suptitle('Correlated Assets Simulation Dashboard', fontsize=20, y=0.98)
    
    # Load and place images
    # 1. Single Path (top left)
    ax1 = fig.add_subplot(grid[0, 0])
    img1 = mpimg.imread('correlated_single_path.png')
    ax1.imshow(img1)
    ax1.set_title('Single Path Simulation')
    ax1.axis('off')
    
    # 2. Correlations (top right)
    ax2 = fig.add_subplot(grid[0, 1])
    img2 = mpimg.imread('correlated_correlations.png')
    ax2.imshow(img2)
    ax2.set_title('Correlation Analysis')
    ax2.axis('off')
    
    # 3. Multiple Paths (middle row, span both columns)
    ax3 = fig.add_subplot(grid[1, :])
    img3 = mpimg.imread('correlated_multiple_paths.png')
    ax3.imshow(img3)
    ax3.set_title('Multiple Paths Simulation')
    ax3.axis('off')
    
    # 4. Portfolio Performance (bottom left)
    ax4 = fig.add_subplot(grid[2, 0])
    img4 = mpimg.imread('correlated_portfolio.png')
    ax4.imshow(img4)
    ax4.set_title('Portfolio Performance')
    ax4.axis('off')
    
    # 5. Portfolio Distribution (bottom right)
    ax5 = fig.add_subplot(grid[2, 1])
    img5 = mpimg.imread('correlated_portfolio_distribution.png')
    ax5.imshow(img5)
    ax5.set_title('Final Portfolio Value Distribution')
    ax5.axis('off')
    
    # Add text with summary information
    summary_text = """
    Correlated Assets Simulation Summary:
    - Three assets modeled with GBM: Asset A (μ=0.08, σ=0.20, S₀=$100), Asset B (μ=0.12, σ=0.30, S₀=$50), Asset C (μ=0.10, σ=0.25, S₀=$75)
    - Correlation structure: Asset A-B (0.60), Asset A-C (0.40), Asset B-C (0.70)
    - Portfolio allocation: 40% Asset A, 30% Asset B, 30% Asset C
    - Simulation shows interdependence of assets and effect on portfolio performance
    - Correlation matrices confirm the simulation correctly reproduces the desired correlation structure
    """
    fig.text(0.5, 0.03, summary_text, ha='center', va='bottom', fontsize=10)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save the dashboard
    plt.savefig('correlated_assets_dashboard.png', dpi=150)
    print("Dashboard created and saved as 'correlated_assets_dashboard.png'")
    plt.close()

if __name__ == "__main__":
    if check_plots():
        create_dashboard()
    else:
        print("Could not create dashboard due to missing plots.") 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def setup_plotting_style():
    """
    Set up matplotlib and seaborn plotting style for consistent visuals.
    """
    # Set seaborn style
    sns.set_style('whitegrid')
    
    # Set matplotlib parameters
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    # Set color palette
    sns.set_palette('muted')

def plot_paths(t, paths, title=None, xlabel='Time', ylabel='Value', figsize=(10, 6),
              legend=True, legend_labels=None, alpha=0.7, mean_line=True, ci=False):
    """
    Plot simulation paths.
    
    Args:
        t (array-like): Time points
        paths (array-like): Simulation paths (2D array where rows are paths)
        title (str, optional): Plot title
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        figsize (tuple): Figure size
        legend (bool): Whether to show legend
        legend_labels (list, optional): Custom legend labels
        alpha (float): Alpha (transparency) for path lines
        mean_line (bool): Whether to plot mean path
        ci (bool or float): Whether to plot confidence interval (can be True for 95% or a float value)
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to 2D array if input is 1D
    if paths.ndim == 1:
        paths = paths.reshape(1, -1)
    
    n_paths = paths.shape[0]
    
    # Plot individual paths
    if n_paths <= 10 or not ci:  # Only plot individual paths if not too many or if CI is off
        for i in range(min(n_paths, 50)):  # Limit to 50 paths to avoid clutter
            if legend_labels is not None and i < len(legend_labels):
                label = legend_labels[i]
            elif i == 0:
                label = 'Sample Path' if n_paths > 1 else 'Path'
            else:
                label = None
            
            ax.plot(t, paths[i], alpha=alpha, label=label)
    
    # Plot mean path
    if mean_line and n_paths > 1:
        mean_path = np.mean(paths, axis=0)
        ax.plot(t, mean_path, 'k-', linewidth=2, label='Mean Path')
    
    # Plot confidence interval
    if ci and n_paths > 1:
        if isinstance(ci, bool):
            ci_level = 0.95  # Default 95% CI
        else:
            ci_level = ci
        
        alpha = (1 + ci_level) / 2
        lower_percentile = (1 - ci_level) / 2 * 100
        upper_percentile = (1 - (1 - ci_level) / 2) * 100
        
        lower_bound = np.percentile(paths, lower_percentile, axis=0)
        upper_bound = np.percentile(paths, upper_percentile, axis=0)
        
        ax.fill_between(t, lower_bound, upper_bound, alpha=0.2, 
                        label=f'{int(ci_level*100)}% Confidence Interval')
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Add legend
    if legend:
        ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig, ax

def plot_distribution(data, bins=30, kde=True, title=None, xlabel=None, 
                      theoretical_dist=None, figsize=(10, 6), ax=None):
    """
    Plot the distribution of data with optional theoretical distribution overlay.
    
    Args:
        data (array-like): Data points to plot
        bins (int): Number of histogram bins
        kde (bool): Whether to plot kernel density estimate
        title (str, optional): Plot title
        xlabel (str, optional): x-axis label
        theoretical_dist (tuple, optional): (dist_name, params) for theoretical distribution
        figsize (tuple): Figure size
        ax (matplotlib.axes, optional): Existing axes to plot on
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Plot histogram and KDE
    sns.histplot(data, bins=bins, kde=kde, ax=ax)
    
    # Add theoretical distribution if provided
    if theoretical_dist is not None:
        dist_name, params = theoretical_dist
        
        if dist_name.lower() == 'normal':
            mu, sigma = params
            x = np.linspace(np.min(data), np.max(data), 1000)
            pdf = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, pdf * len(data) * (x[1] - x[0]), 'r-', linewidth=2, 
                   label=f'Normal({mu:.4f}, {sigma:.4f})')
        
        elif dist_name.lower() == 'lognormal':
            mu, sigma = params
            x = np.linspace(np.max(0.0001, np.min(data)), np.max(data), 1000)
            pdf = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
            ax.plot(x, pdf * len(data) * (x[1] - x[0]), 'r-', linewidth=2,
                   label=f'LogNormal({mu:.4f}, {sigma:.4f})')
        
        ax.legend()
    
    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    
    return fig, ax

def plot_calibration_results(true_params, estimated_params, param_names=None, figsize=(10, 6)):
    """
    Plot comparison of true vs. estimated parameters.
    
    Args:
        true_params (array-like): True parameter values
        estimated_params (array-like): Estimated parameter values
        param_names (list, optional): Parameter names
        figsize (tuple): Figure size
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    if param_names is None:
        param_names = [f'Param {i+1}' for i in range(len(true_params))]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(true_params))
    width = 0.35
    
    ax.bar(x - width/2, true_params, width, label='True')
    ax.bar(x + width/2, estimated_params, width, label='Estimated')
    
    ax.set_xticks(x)
    ax.set_xticklabels(param_names)
    ax.legend()
    
    ax.set_title('Parameter Estimation Results')
    ax.grid(axis='y', alpha=0.3)
    
    # Add text with relative error
    for i, (true, est) in enumerate(zip(true_params, estimated_params)):
        rel_error = abs(est - true) / abs(true) * 100 if true != 0 else abs(est - true)
        ax.text(i, max(true, est) + 0.05 * max(abs(true_params)), 
                f'Error: {rel_error:.2f}%', ha='center')
    
    return fig, ax

def plot_error_convergence(dt_values, errors, method_names=None, log_scale=True, 
                          figsize=(10, 6), title=None):
    """
    Plot error convergence as a function of time step size.
    
    Args:
        dt_values (array-like): Time step sizes
        errors (dict or array-like): Error values for each method
        method_names (list, optional): Method names
        log_scale (bool): Whether to use log scale for both axes
        figsize (tuple): Figure size
        title (str, optional): Plot title
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle different input formats
    if isinstance(errors, dict):
        for i, (method_name, method_errors) in enumerate(errors.items()):
            if isinstance(method_errors, list) and isinstance(method_errors[0], dict):
                # Handle case where errors is a dict of lists of dicts
                y_values = [e['mean'] for e in method_errors]
                ax.plot(dt_values, y_values, 'o-', label=method_name)
            else:
                # Handle case where errors is a dict of arrays
                ax.plot(dt_values, method_errors, 'o-', label=method_name)
    else:
        # Handle case where errors is a 2D array (methods x dt_values)
        for i, method_errors in enumerate(errors):
            label = method_names[i] if method_names and i < len(method_names) else f'Method {i+1}'
            ax.plot(dt_values, method_errors, 'o-', label=label)
    
    # Set log scale if requested
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # Set labels and title
    ax.set_xlabel('Time Step Size (dt)')
    ax.set_ylabel('Error')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Convergence of Numerical Methods')
    
    # Add reference lines
    x_min, x_max = ax.get_xlim()
    x_ref = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    
    # O(dt) reference line (first-order convergence)
    y_ref_1 = x_ref
    y_scale_1 = 0.5 * ax.get_ylim()[1] / y_ref_1[-1]
    ax.plot(x_ref, y_ref_1 * y_scale_1, 'k--', alpha=0.5, label='O(dt)')
    
    # O(dt^2) reference line (second-order convergence)
    y_ref_2 = x_ref**2
    y_scale_2 = 0.5 * ax.get_ylim()[1] / y_ref_2[-1]
    ax.plot(x_ref, y_ref_2 * y_scale_2, 'k:', alpha=0.5, label='O(dt²)')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax

def plot_qq(data, dist='norm', title=None, figsize=(8, 8)):
    """
    Create a Q-Q plot to compare data with a theoretical distribution.
    
    Args:
        data (array-like): Data points to plot
        dist (str): Distribution name ('norm', 'lognorm', etc.)
        title (str, optional): Plot title
        figsize (tuple): Figure size
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create Q-Q plot
    stats.probplot(data, dist=dist, plot=ax)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Q-Q Plot ({dist} distribution)')
    
    return fig, ax 
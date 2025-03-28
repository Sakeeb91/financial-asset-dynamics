import numpy as np
import pandas as pd
from ..models.gbm import GeometricBrownianMotion

def estimate_gbm_params(prices, dt=1/252, annualize=True):
    """
    Estimate the parameters of a Geometric Brownian Motion model from price data.
    
    The GBM SDE is:
    dS_t = μ*S_t dt + σ*S_t dW_t
    
    Args:
        prices (array-like): Time series of asset prices
        dt (float): Time step size (default is 1/252 for daily data assuming 252 trading days)
        annualize (bool): Whether to annualize the parameters
    
    Returns:
        tuple: (mu, sigma) estimated drift and volatility parameters
    """
    # Calculate log returns
    log_returns = np.diff(np.log(prices))
    
    # Estimate parameters
    mu_hat = np.mean(log_returns) / dt
    sigma_hat = np.std(log_returns) / np.sqrt(dt)
    
    # If not annualizing, return as is
    if not annualize:
        return mu_hat, sigma_hat
    
    # Calculate annualized values
    if dt != 1:
        trading_days = 1 / dt
        mu_annual = mu_hat * trading_days
        sigma_annual = sigma_hat * np.sqrt(trading_days)
        return mu_annual, sigma_annual
    else:
        return mu_hat, sigma_hat

def create_gbm_model(prices, dt=1/252, annualize=True):
    """
    Create a calibrated GBM model from price data.
    
    Args:
        prices (array-like): Time series of asset prices
        dt (float): Time step size
        annualize (bool): Whether to annualize the parameters
    
    Returns:
        GeometricBrownianMotion: Calibrated GBM model
    """
    mu, sigma = estimate_gbm_params(prices, dt, annualize)
    return GeometricBrownianMotion(mu, sigma)

def compute_log_likelihood(prices, mu, sigma, dt=1/252):
    """
    Compute the log-likelihood of GBM parameters given price data.
    
    The log-likelihood function for GBM can be derived from the fact that
    log returns are normally distributed with mean (μ - σ²/2)*dt and variance σ²*dt.
    
    Args:
        prices (array-like): Time series of asset prices
        mu (float): Drift parameter
        sigma (float): Volatility parameter
        dt (float): Time step size
    
    Returns:
        float: Log-likelihood value
    """
    # Calculate log returns
    log_returns = np.diff(np.log(prices))
    n = len(log_returns)
    
    # Mean and variance of log returns
    mean_returns = (mu - 0.5 * sigma**2) * dt
    var_returns = sigma**2 * dt
    
    # Log-likelihood for normal distribution
    log_lik = -0.5 * n * np.log(2 * np.pi * var_returns) - \
              0.5 * np.sum((log_returns - mean_returns)**2) / var_returns
    
    return log_lik

def maximum_likelihood_estimation(prices, dt=1/252, annualize=True):
    """
    Perform Maximum Likelihood Estimation for GBM parameters.
    
    For GBM, the MLE solution is actually the same as the simple method
    in estimate_gbm_params, but this function also returns confidence intervals.
    
    Args:
        prices (array-like): Time series of asset prices
        dt (float): Time step size
        annualize (bool): Whether to annualize the parameters
    
    Returns:
        dict: Dictionary with parameter estimates and confidence intervals
    """
    log_returns = np.diff(np.log(prices))
    n = len(log_returns)
    
    # Parameter estimates (MLE)
    mu_hat = np.mean(log_returns) / dt + 0.5 * np.var(log_returns) / dt
    sigma_hat = np.sqrt(np.var(log_returns) / dt)
    
    # Standard errors (asymptotic)
    se_mu = sigma_hat / np.sqrt(n * dt)
    se_sigma = sigma_hat / np.sqrt(2 * n)
    
    # Annualize if requested
    if annualize and dt != 1:
        trading_days = 1 / dt
        mu_hat = mu_hat * trading_days
        sigma_hat = sigma_hat * np.sqrt(trading_days)
        se_mu = se_mu * trading_days
        se_sigma = se_sigma * np.sqrt(trading_days)
    
    # 95% confidence intervals
    mu_ci = (mu_hat - 1.96 * se_mu, mu_hat + 1.96 * se_mu)
    sigma_ci = (sigma_hat - 1.96 * se_sigma, sigma_hat + 1.96 * se_sigma)
    
    # Calculate the log-likelihood
    log_lik = compute_log_likelihood(prices, mu_hat / trading_days if annualize else mu_hat,
                                    sigma_hat / np.sqrt(trading_days) if annualize else sigma_hat, 
                                    dt)
    
    return {
        'mu': mu_hat,
        'sigma': sigma_hat,
        'mu_std_error': se_mu,
        'sigma_std_error': se_sigma,
        'mu_conf_interval': mu_ci,
        'sigma_conf_interval': sigma_ci,
        'log_likelihood': log_lik,
        'n_observations': n
    } 
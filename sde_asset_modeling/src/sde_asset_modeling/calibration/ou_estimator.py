import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from ..models.ou import OrnsteinUhlenbeck

def estimate_ou_params_regression(data, dt=1/252, annualize=True):
    """
    Estimate the parameters of an Ornstein-Uhlenbeck process using regression.
    
    The OU SDE is:
    dX_t = θ(μ - X_t) dt + σ dW_t
    
    This uses the discrete-time approximation:
    X_{t+dt} - X_t = θ(μ - X_t)dt + ε_t
    where ε_t ~ N(0, σ²dt)
    
    This can be rewritten as a linear regression:
    ΔX_t = a + b X_t + ε_t
    where a = θμdt and b = -θdt
    
    Args:
        data (array-like): Time series of data
        dt (float): Time step size (default is 1/252 for daily data)
        annualize (bool): Whether to annualize the parameters
    
    Returns:
        tuple: (theta, mu, sigma) estimated parameters
    """
    # Calculate differences
    x = np.array(data[:-1])
    y = np.diff(data)
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Extract OU parameters
    theta_hat = -slope / dt
    mu_hat = intercept / (theta_hat * dt)
    
    # Estimate volatility from residuals
    y_pred = intercept + slope * x
    residuals = y - y_pred
    sigma_hat = np.std(residuals) / np.sqrt(dt)
    
    # If not annualizing, return as is
    if not annualize:
        return theta_hat, mu_hat, sigma_hat
    
    # Calculate annualized values
    if dt != 1:
        trading_days = 1 / dt
        theta_annual = theta_hat  # Mean reversion speed is already in units of time^-1
        sigma_annual = sigma_hat * np.sqrt(trading_days)
        return theta_annual, mu_hat, sigma_annual
    else:
        return theta_hat, mu_hat, sigma_hat

def create_ou_model(data, dt=1/252, annualize=True, method='regression'):
    """
    Create a calibrated OU model from data.
    
    Args:
        data (array-like): Time series of data
        dt (float): Time step size
        annualize (bool): Whether to annualize the parameters
        method (str): Estimation method ('regression' or 'mle')
    
    Returns:
        OrnsteinUhlenbeck: Calibrated OU model
    """
    if method.lower() == 'regression':
        theta, mu, sigma = estimate_ou_params_regression(data, dt, annualize)
    elif method.lower() == 'mle':
        params = maximum_likelihood_estimation(data, dt, annualize)
        theta, mu, sigma = params['theta'], params['mu'], params['sigma']
    else:
        raise ValueError(f"Unknown estimation method: {method}")
    
    return OrnsteinUhlenbeck(theta, mu, sigma)

def ou_log_likelihood(params, data, dt):
    """
    Compute the log-likelihood for OU process parameters.
    
    Args:
        params (array-like): [theta, mu, sigma] parameters
        data (array-like): Time series of data
        dt (float): Time step size
    
    Returns:
        float: Negative log-likelihood value (for minimization)
    """
    theta, mu, sigma = params
    
    if theta <= 0 or sigma <= 0:
        return 1e10  # Return a large value for invalid parameters
    
    n = len(data) - 1
    x_t = data[:-1]
    x_tp1 = data[1:]
    
    # Mean and variance of the conditional distribution X_{t+dt} | X_t
    exp_term = np.exp(-theta * dt)
    cond_mean = x_t * exp_term + mu * (1 - exp_term)
    cond_var = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt))
    
    # Log-likelihood calculation
    log_lik = -0.5 * n * np.log(2 * np.pi * cond_var) - \
              0.5 * np.sum((x_tp1 - cond_mean)**2) / cond_var
    
    return -log_lik  # Return negative for minimization

def maximum_likelihood_estimation(data, dt=1/252, annualize=True):
    """
    Perform Maximum Likelihood Estimation for OU parameters.
    
    Args:
        data (array-like): Time series of data
        dt (float): Time step size
        annualize (bool): Whether to annualize the parameters
    
    Returns:
        dict: Dictionary with parameter estimates
    """
    # Initial estimates from regression method
    theta_init, mu_init, sigma_init = estimate_ou_params_regression(data, dt, False)
    
    # MLE optimization
    result = minimize(
        ou_log_likelihood,
        [theta_init, mu_init, sigma_init],
        args=(data, dt),
        bounds=[(1e-6, None), (None, None), (1e-6, None)],
        method='L-BFGS-B'
    )
    
    if not result.success:
        print(f"Warning: MLE optimization did not converge: {result.message}")
    
    theta_hat, mu_hat, sigma_hat = result.x
    
    # Calculate standard errors using observed Fisher information
    # (this is a simplified approach; a more accurate method would use the Hessian)
    n = len(data) - 1
    se_theta = np.sqrt(2 * theta_hat**2 / n)
    se_mu = np.sqrt(sigma_hat**2 / (2 * n * theta_hat))
    se_sigma = np.sqrt(sigma_hat**2 / (2 * n))
    
    # Annualize if requested
    if annualize and dt != 1:
        trading_days = 1 / dt
        sigma_hat = sigma_hat * np.sqrt(trading_days)
        se_sigma = se_sigma * np.sqrt(trading_days)
    
    # 95% confidence intervals
    theta_ci = (theta_hat - 1.96 * se_theta, theta_hat + 1.96 * se_theta)
    mu_ci = (mu_hat - 1.96 * se_mu, mu_hat + 1.96 * se_mu)
    sigma_ci = (sigma_hat - 1.96 * se_sigma, sigma_hat + 1.96 * se_sigma)
    
    return {
        'theta': theta_hat,
        'mu': mu_hat,
        'sigma': sigma_hat,
        'theta_std_error': se_theta,
        'mu_std_error': se_mu,
        'sigma_std_error': se_sigma,
        'theta_conf_interval': theta_ci,
        'mu_conf_interval': mu_ci,
        'sigma_conf_interval': sigma_ci,
        'log_likelihood': -result.fun,
        'n_observations': n,
        'converged': result.success
    } 
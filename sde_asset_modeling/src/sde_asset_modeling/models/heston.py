"""
Heston Stochastic Volatility Model
This module implements the Heston model for asset prices with stochastic volatility.
"""

import numpy as np
from .base_sde import BaseSDE


class HestonModel(BaseSDE):
    """
    Heston Stochastic Volatility Model for asset prices.
    
    The process follows a system of two SDEs:
    
    dS = μS dt + √v S dW_1
    dv = κ(θ - v) dt + ξ√v dW_2
    
    where:
    - S is the asset price
    - v is the variance (volatility squared)
    - μ is the drift rate
    - κ is the rate of mean reversion for variance
    - θ is the long-term mean of variance
    - ξ is the volatility of volatility
    - dW_1, dW_2 are Wiener processes with correlation ρ
    """
    
    def __init__(self, mu, kappa, theta, xi, rho, s0, v0):
        """
        Initialize Heston model.
        
        Args:
            mu (float): Drift coefficient (annual expected return)
            kappa (float): Rate of mean reversion for variance
            theta (float): Long-term mean of variance
            xi (float): Volatility of volatility
            rho (float): Correlation between price and volatility Wiener processes
            s0 (float): Initial asset price
            v0 (float): Initial variance (volatility squared)
        """
        # Call parent constructor with placeholder sigma (not directly used)
        super().__init__(mu, np.sqrt(theta), s0)
        
        self.kappa = kappa  # Mean reversion rate
        self.theta = theta  # Long-term variance
        self.xi = xi  # Volatility of volatility
        self.rho = rho  # Correlation between stock and variance processes
        self.v0 = v0  # Initial variance
        
    def get_parameters(self):
        """Get all model parameters."""
        return {
            'mu': self.mu,
            'kappa': self.kappa,
            'theta': self.theta,
            'xi': self.xi,
            'rho': self.rho,
            'initial_price': self.x0,
            'initial_variance': self.v0
        }
    
    def drift(self, x, t, v=None):
        """
        Drift coefficient function for the price process.
        
        Args:
            x (float): Current asset price
            t (float): Current time
            v (float, optional): Current variance
            
        Returns:
            float: Drift coefficient value
        """
        return self.mu * x
    
    def diffusion(self, x, t, v=None):
        """
        Diffusion coefficient function for the price process.
        
        Args:
            x (float): Current asset price
            t (float): Current time
            v (float, optional): Current variance
            
        Returns:
            float: Diffusion coefficient value
        """
        # If variance is provided, use it; otherwise use the long-term variance
        vol = np.sqrt(v) if v is not None else np.sqrt(self.theta)
        return vol * x
    
    def variance_drift(self, v, t):
        """
        Drift coefficient function for the variance process.
        
        Args:
            v (float): Current variance
            t (float): Current time
            
        Returns:
            float: Variance drift coefficient value
        """
        return self.kappa * (self.theta - v)
    
    def variance_diffusion(self, v, t):
        """
        Diffusion coefficient function for the variance process.
        
        Args:
            v (float): Current variance
            t (float): Current time
            
        Returns:
            float: Variance diffusion coefficient value
        """
        # Max ensures variance stays positive for numerical stability
        return self.xi * np.sqrt(max(1e-10, v))
    
    def exact_solution(self, t, W_S, W_v, Z=None):
        """
        There is no closed-form solution for sample paths of the Heston model.
        This is a placeholder function that returns None.
        """
        raise NotImplementedError("Exact solution for Heston model paths is not available")
    
    def generate_correlated_wiener_processes(self, n_steps, dt, n_paths=1):
        """
        Generate correlated Wiener processes for price and variance.
        
        Args:
            n_steps (int): Number of time steps
            dt (float): Time step size
            n_paths (int): Number of paths to generate
            
        Returns:
            tuple: (W_S, W_v) correlated Wiener processes
        """
        # Generate independent standard normal random variables
        Z1 = np.random.normal(size=(n_paths, n_steps))
        Z2 = np.random.normal(size=(n_paths, n_steps))
        
        # Create correlated random variables using Cholesky decomposition
        # W_S = Z1
        # W_v = rho * Z1 + sqrt(1-rho^2) * Z2
        W_S = Z1 * np.sqrt(dt)
        W_v = (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2) * np.sqrt(dt)
        
        return W_S, W_v 
"""
SABR (Stochastic Alpha Beta Rho) Model
This module implements the SABR model for asset prices, commonly used for interest rate derivatives.
"""

import numpy as np
from .base_sde import BaseSDE


class SABRModel(BaseSDE):
    """
    SABR (Stochastic Alpha Beta Rho) Model for interest rate derivatives.
    
    The process follows a system of two SDEs:
    
    dF = α F^β dW_1
    dα = ν α dW_2
    
    where:
    - F is the forward rate
    - α is the stochastic volatility
    - β is the CEV (constant elasticity of variance) parameter
    - ν is the volatility of volatility
    - dW_1, dW_2 are Wiener processes with correlation ρ
    """
    
    def __init__(self, alpha, beta, nu, rho, f0, alpha0):
        """
        Initialize SABR model.
        
        Args:
            alpha (float): Initial volatility level
            beta (float): CEV parameter (0 <= beta <= 1)
            nu (float): Volatility of volatility
            rho (float): Correlation between rate and volatility processes
            f0 (float): Initial forward rate
            alpha0 (float): Initial volatility
        """
        # Initialize base parameters directly instead of using super()
        self.mu = 0.0  # Not used in SABR
        self.sigma = alpha  # Initial volatility parameter
        self.x0 = f0  # Initial forward rate
        
        self.beta = beta  # CEV parameter
        self.nu = nu  # Volatility of volatility
        self.rho = rho  # Correlation
        self.alpha0 = alpha0  # Initial volatility
        
    def get_parameters(self):
        """Get all model parameters."""
        return {
            'alpha': self.sigma,  # Using sigma from parent class as alpha
            'beta': self.beta,
            'nu': self.nu,
            'rho': self.rho,
            'initial_forward': self.x0,
            'initial_alpha': self.alpha0
        }
    
    def drift(self, f, t, alpha=None):
        """
        Drift coefficient function for the forward rate process.
        For SABR, the drift is 0 in the risk-neutral measure.
        
        Args:
            f (float): Current forward rate
            t (float): Current time
            alpha (float, optional): Current volatility
            
        Returns:
            float: Drift coefficient value (0 for SABR)
        """
        return 0.0
    
    def diffusion(self, f, t, alpha=None):
        """
        Diffusion coefficient function for the forward rate process.
        
        Args:
            f (float): Current forward rate
            t (float): Current time
            alpha (float, optional): Current volatility
            
        Returns:
            float: Diffusion coefficient value
        """
        # If alpha is provided, use it; otherwise use the initial alpha
        vol = alpha if alpha is not None else self.sigma
        
        # Handle the case where f is very close to zero for β > 0
        if f < 1e-10 and self.beta > 0:
            return vol * (1e-10)**self.beta
        
        # Standard SABR diffusion term
        return vol * abs(f)**self.beta
    
    def alpha_drift(self, alpha, t):
        """
        Drift coefficient function for the volatility process.
        
        Args:
            alpha (float): Current volatility
            t (float): Current time
            
        Returns:
            float: Volatility drift coefficient value (0 for log-normal volatility)
        """
        return 0.0
    
    def alpha_diffusion(self, alpha, t):
        """
        Diffusion coefficient function for the volatility process.
        
        Args:
            alpha (float): Current volatility
            t (float): Current time
            
        Returns:
            float: Volatility diffusion coefficient value
        """
        return self.nu * alpha
    
    def sabr_implied_volatility(self, K, T):
        """
        Calculate the Black implied volatility using the SABR approximation formula.
        
        This is the Hagan et al. (2002) formula for SABR implied volatility.
        
        Args:
            K (float): Strike price
            T (float): Time to expiry
            
        Returns:
            float: Black implied volatility
        """
        F = self.x0  # Forward rate
        alpha = self.alpha0  # Initial volatility
        beta = self.beta  # CEV parameter
        nu = self.nu  # Volatility of volatility
        rho = self.rho  # Correlation
        
        # Handle ATM case separately to avoid numerical issues
        if abs(F - K) < 1e-10:
            # ATM formula
            return (alpha / (F**(1-beta))) * (
                1 + 
                ((1-beta)**2/24) * alpha**2/(F**(2-2*beta)) * T +
                (1/4) * (rho*beta*nu*alpha)/(F**(1-beta)) * T +
                ((2-3*rho**2)/24) * nu**2 * T
            )
        
        # For non-ATM cases
        z = (nu/alpha) * (F*K)**((1-beta)/2) * np.log(F/K)
        chi = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
        
        # SABR implied volatility formula
        return (nu * (F*K)**((1-beta)/2) * z / chi) * (
            1 + 
            ((1-beta)**2/24) * np.log(F/K)**2 +
            ((1-beta)**4/1920) * np.log(F/K)**4 +
            (1/24) * (rho*beta*nu*alpha)/((F*K)**((1-beta)/2)) * T +
            ((2-3*rho**2)/24) * nu**2 * T
        )
    
    def generate_correlated_wiener_processes(self, n_steps, dt, n_paths=1):
        """
        Generate correlated Wiener processes for forward rate and volatility.
        
        Args:
            n_steps (int): Number of time steps
            dt (float): Time step size
            n_paths (int): Number of paths to generate
            
        Returns:
            tuple: (W_F, W_alpha) correlated Wiener processes
        """
        # Generate independent standard normal random variables
        Z1 = np.random.normal(size=(n_paths, n_steps))
        Z2 = np.random.normal(size=(n_paths, n_steps))
        
        # Create correlated random variables using the correlation coefficient
        # W_F = Z1
        # W_alpha = rho * Z1 + sqrt(1-rho^2) * Z2
        W_F = Z1 * np.sqrt(dt)
        W_alpha = (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2) * np.sqrt(dt)
        
        return W_F, W_alpha
    
    def exact_solution(self, t, W_F, W_alpha, Z=None):
        """
        There is no closed-form solution for sample paths of the SABR model.
        This is a placeholder function that raises an error.
        """
        raise NotImplementedError("Exact solution for SABR model paths is not available") 
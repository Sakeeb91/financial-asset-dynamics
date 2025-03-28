"""
Jump Diffusion (Merton) Model
This module implements the Jump Diffusion model for asset prices.
"""

import numpy as np
from .base_sde import BaseSDE


class MertonJumpDiffusion(BaseSDE):
    """
    Merton Jump Diffusion Model for asset prices.
    
    The process follows:
    dS = μS dt + σS dW + S dJ
    
    where:
    - μ is the drift rate
    - σ is the volatility
    - dW is the Wiener process increment
    - dJ is the compound Poisson process for jumps
    
    The jumps are modeled as:
    dJ = (e^Y - 1) dN
    
    where:
    - Y ~ N(jump_mean, jump_std) is the jump size (log of the price ratio)
    - dN is a Poisson process with intensity lambda_j (jump frequency)
    """
    
    def __init__(self, mu, sigma, lambda_j, jump_mean, jump_std, x0):
        """
        Initialize Merton Jump Diffusion model.
        
        Args:
            mu (float): Drift coefficient (annual expected return)
            sigma (float): Diffusion coefficient (annual volatility)
            lambda_j (float): Jump intensity (expected number of jumps per year)
            jump_mean (float): Mean of the jump size distribution (log of the price ratio)
            jump_std (float): Standard deviation of the jump size
            x0 (float): Initial asset price
        """
        # Initialize base parameters directly instead of using super()
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0
        
        self.lambda_j = lambda_j  # Jump frequency
        self.jump_mean = jump_mean  # Mean jump size (log scale)
        self.jump_std = jump_std  # Jump size volatility
        
        # Adjust drift to ensure risk-neutral pricing
        # The expected value of e^Y - 1 is e^(jump_mean + jump_std^2/2) - 1
        self.jump_expected = np.exp(jump_mean + jump_std**2/2) - 1
        
        # Effective drift is reduced by the jump component's expected contribution
        self.mu_effective = mu - lambda_j * self.jump_expected
        
    def get_parameters(self):
        """Get all model parameters."""
        return {
            'mu': self.mu,
            'sigma': self.sigma,
            'lambda_j': self.lambda_j,
            'jump_mean': self.jump_mean,
            'jump_std': self.jump_std,
            'initial_value': self.x0
        }
    
    def drift(self, x, t):
        """
        Drift coefficient function.
        
        Args:
            x (float): Current asset price
            t (float): Current time
            
        Returns:
            float: Drift coefficient value
        """
        return self.mu_effective * x
    
    def diffusion(self, x, t):
        """
        Diffusion coefficient function.
        
        Args:
            x (float): Current asset price
            t (float): Current time
            
        Returns:
            float: Diffusion coefficient value
        """
        return self.sigma * x
    
    def generate_jumps(self, t_span, dt):
        """
        Generate jump times and sizes for the simulation period.
        
        Args:
            t_span (tuple): (t_start, t_end) time interval
            dt (float): Time step size
            
        Returns:
            tuple: (jump_times, jump_sizes)
                - jump_times: Array of times when jumps occur
                - jump_sizes: Array of corresponding jump sizes (as ratios)
        """
        t_start, t_end = t_span
        total_time = t_end - t_start
        
        # Expected number of jumps in the interval
        expected_jumps = self.lambda_j * total_time
        
        # Generate actual number of jumps (Poisson distributed)
        n_jumps = np.random.poisson(expected_jumps)
        
        if n_jumps == 0:
            return np.array([]), np.array([])
        
        # Generate jump times (uniformly distributed in the interval)
        jump_times = np.random.uniform(t_start, t_end, n_jumps)
        jump_times.sort()  # Sort jump times in ascending order
        
        # Generate jump sizes (log-normally distributed)
        jump_log_sizes = np.random.normal(self.jump_mean, self.jump_std, n_jumps)
        jump_sizes = np.exp(jump_log_sizes) - 1  # Convert to relative price changes
        
        return jump_times, jump_sizes
    
    def exact_solution(self, t, W, Z=None, jump_times=None, jump_sizes=None):
        """
        Compute the exact solution of the SDE for GBM with jumps.
        
        Args:
            t (float or array): Time points
            W (float or array): Brownian motion values at time t
            Z (float or array, optional): Additional random process (unused)
            jump_times (array, optional): Times when jumps occur
            jump_sizes (array, optional): Sizes of the jumps (as ratios)
            
        Returns:
            float or array: Solution of the SDE at time t
        """
        # If t is a scalar, convert to array for uniform processing
        scalar_input = np.isscalar(t)
        t = np.atleast_1d(t)
        W = np.atleast_1d(W)
        
        # Continuous GBM component
        # S(t) = S(0) * exp((mu - sigma^2/2)*t + sigma*W(t))
        continuous_part = self.x0 * np.exp((self.mu_effective - 0.5 * self.sigma**2) * t + self.sigma * W)
        
        # Handle the jump component if provided
        if jump_times is not None and jump_sizes is not None and len(jump_times) > 0:
            # Calculate the product of jump factors for each time point
            jump_factors = np.ones_like(t)
            
            for i, current_t in enumerate(t):
                # Find all jumps that occurred before or at the current time
                jumps_before = jump_sizes[jump_times <= current_t]
                
                if len(jumps_before) > 0:
                    # Multiply by all jump factors (1 + jump_size)
                    jump_factors[i] = np.prod(1 + jumps_before)
            
            # Apply jump factors to the continuous part
            result = continuous_part * jump_factors
        else:
            result = continuous_part
        
        # Return scalar if input was scalar
        return result[0] if scalar_input else result 
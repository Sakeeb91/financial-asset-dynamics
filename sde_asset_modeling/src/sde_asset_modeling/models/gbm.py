import numpy as np
from .base_sde import BaseSDE

class GeometricBrownianMotion(BaseSDE):
    """
    Geometric Brownian Motion (GBM) SDE model.
    
    dS_t = μ*S_t dt + σ*S_t dW_t
    
    where:
    - μ (mu) is the drift coefficient (expected return)
    - σ (sigma) is the volatility coefficient
    - S_t is the asset price at time t
    - dW_t is the increment of a Wiener process
    
    GBM is commonly used to model stock prices and other financial assets.
    """
    
    def __init__(self, mu, sigma):
        """
        Initialize the GBM model.
        
        Args:
            mu (float): Drift coefficient (expected return)
            sigma (float): Volatility coefficient
        """
        self.mu = mu
        self.sigma = sigma
        
    def drift(self, x, t):
        """
        Drift function for GBM: μ*S_t
        
        Args:
            x (float): Current asset price S_t
            t (float): Current time t (not used in standard GBM)
            
        Returns:
            float: Drift term value
        """
        return self.mu * x
    
    def diffusion(self, x, t):
        """
        Diffusion function for GBM: σ*S_t
        
        Args:
            x (float): Current asset price S_t
            t (float): Current time t (not used in standard GBM)
            
        Returns:
            float: Diffusion term value
        """
        return self.sigma * x
    
    def diffusion_derivative(self, x, t):
        """
        Derivative of diffusion function for GBM: σ
        Analytical solution available for GBM.
        
        Args:
            x (float): Current asset price S_t
            t (float): Current time t (not used)
            
        Returns:
            float: Derivative of diffusion with respect to x
        """
        return self.sigma
    
    def exact_solution(self, s0, t, random_state=None):
        """
        Generate samples from the exact solution of GBM.
        
        The analytical solution of GBM is:
        S_t = S_0 * exp((μ - σ²/2)*t + σ*W_t)
        
        Args:
            s0 (float): Initial asset price S_0
            t (float or array): Time point(s) to generate solution for
            random_state (int, optional): Seed for random number generator
            
        Returns:
            float or array: Asset price at time(s) t
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        # Handle both scalar t and array-like t
        t = np.asarray(t)
        scalar_input = False
        if t.ndim == 0:
            t = t[np.newaxis]
            scalar_input = True
            
        # Generate Wiener process increments
        dw = np.random.normal(0, np.sqrt(t))
        
        # Calculate analytical solution
        s_t = s0 * np.exp((self.mu - 0.5 * self.sigma**2) * t + self.sigma * dw)
        
        return s_t[0] if scalar_input else s_t 
import numpy as np
from .base_sde import BaseSDE

class OrnsteinUhlenbeck(BaseSDE):
    """
    Ornstein-Uhlenbeck (OU) process SDE model.
    
    dX_t = θ(μ - X_t) dt + σ dW_t
    
    where:
    - θ (theta) is the speed of mean reversion
    - μ (mu) is the long-term mean level
    - σ (sigma) is the volatility
    - X_t is the state variable at time t
    - dW_t is the increment of a Wiener process
    
    The OU process is mean-reverting, making it useful for modeling
    interest rates, volatility, and other mean-reverting quantities.
    """
    
    def __init__(self, theta, mu, sigma):
        """
        Initialize the OU process model.
        
        Args:
            theta (float): Speed of mean reversion
            mu (float): Long-term mean level
            sigma (float): Volatility coefficient
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
    
    def drift(self, x, t):
        """
        Drift function for OU process: θ(μ - X_t)
        
        Args:
            x (float): Current state X_t
            t (float): Current time t (not used in standard OU process)
            
        Returns:
            float: Drift term value
        """
        return self.theta * (self.mu - x)
    
    def diffusion(self, x, t):
        """
        Diffusion function for OU process: σ
        
        Args:
            x (float): Current state X_t
            t (float): Current time t (not used in standard OU process)
            
        Returns:
            float: Diffusion term value
        """
        return self.sigma
    
    def diffusion_derivative(self, x, t):
        """
        Derivative of diffusion function for OU: 0
        Since diffusion is constant, its derivative is zero.
        
        Args:
            x (float): Current state X_t
            t (float): Current time t (not used)
            
        Returns:
            float: Derivative of diffusion with respect to x (always 0 for OU)
        """
        return 0.0
    
    def exact_solution(self, x0, t, random_state=None):
        """
        Generate samples from the exact solution of the OU process.
        
        The analytical solution of the OU process is:
        X_t = X_0 * e^(-θt) + μ(1 - e^(-θt)) + σ√(1 - e^(-2θt))/(2θ) * Z
        where Z is a standard normal random variable.
        
        Args:
            x0 (float): Initial state X_0
            t (float or array): Time point(s) to generate solution for
            random_state (int, optional): Seed for random number generator
            
        Returns:
            float or array: State at time(s) t
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        # Handle both scalar t and array-like t
        t = np.asarray(t)
        scalar_input = False
        if t.ndim == 0:
            t = t[np.newaxis]
            scalar_input = True
            
        # Calculate components of the solution
        exp_term = np.exp(-self.theta * t)
        mean = x0 * exp_term + self.mu * (1 - exp_term)
        variance = (self.sigma**2 / (2 * self.theta)) * (1 - np.exp(-2 * self.theta * t))
        
        # Generate samples from the solution
        z = np.random.normal(0, 1, size=t.shape)
        x_t = mean + np.sqrt(variance) * z
        
        return x_t[0] if scalar_input else x_t
    
    def stationary_distribution_parameters(self):
        """
        Return parameters of the stationary distribution of the OU process.
        The stationary distribution is a Normal distribution.
        
        Returns:
            tuple: (mean, variance) of the stationary distribution
        """
        mean = self.mu
        variance = self.sigma**2 / (2 * self.theta)
        return mean, variance 
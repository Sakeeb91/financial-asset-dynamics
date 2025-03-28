import numpy as np
from abc import ABC, abstractmethod

class BaseSDE(ABC):
    """
    Abstract base class for Stochastic Differential Equations (SDEs).
    All specific SDE models should inherit from this class.
    
    The general form of an SDE is:
    dX_t = a(X_t, t) dt + b(X_t, t) dW_t
    
    where:
    - a(X_t, t) is the drift function
    - b(X_t, t) is the diffusion function
    - dW_t is the increment of a Wiener process
    """
    
    @abstractmethod
    def drift(self, x, t):
        """
        Drift function a(x, t) of the SDE.
        
        Args:
            x: Current state
            t: Current time
            
        Returns:
            Drift term value
        """
        pass
    
    @abstractmethod
    def diffusion(self, x, t):
        """
        Diffusion function b(x, t) of the SDE.
        
        Args:
            x: Current state
            t: Current time
            
        Returns:
            Diffusion term value
        """
        pass
    
    def diffusion_derivative(self, x, t):
        """
        Derivative of the diffusion function with respect to x.
        Used in higher-order schemes like Milstein.
        
        Default implementation uses finite difference approximation.
        Override for analytical derivatives if available.
        
        Args:
            x: Current state
            t: Current time
            
        Returns:
            Derivative of diffusion term with respect to x
        """
        # Default implementation using central difference
        h = 1e-6
        return (self.diffusion(x + h, t) - self.diffusion(x - h, t)) / (2 * h) 
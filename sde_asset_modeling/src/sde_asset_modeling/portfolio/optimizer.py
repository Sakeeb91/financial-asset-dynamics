"""
Advanced Portfolio Optimization Module

This module implements modern portfolio theory techniques including:
- Mean-variance optimization (Markowitz)
- Black-Litterman model
- Risk parity
- Maximum diversification
- Minimum variance
- Mean-CVaR optimization
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
from scipy.linalg import block_diag
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    Advanced portfolio optimization with multiple strategies.
    """
    
    def __init__(self, returns_data, risk_free_rate=0.02):
        """
        Initialize the portfolio optimizer.
        
        Args:
            returns_data (pd.DataFrame): Historical returns data (assets as columns)
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculations
        """
        self.returns = returns_data
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns_data.columns)
        self.asset_names = returns_data.columns.tolist()
        
        # Calculate basic statistics
        self.mean_returns = returns_data.mean()
        self.cov_matrix = returns_data.cov()
        self.corr_matrix = returns_data.corr()
        
    def markowitz_optimization(self, target_return=None, min_weight=0.0, max_weight=1.0):
        """
        Classical Markowitz mean-variance optimization.
        
        Args:
            target_return (float, optional): Target portfolio return
            min_weight (float): Minimum weight constraint
            max_weight (float): Maximum weight constraint
            
        Returns:
            dict: Optimization results including weights, expected return, and risk
        """
        # Define variables
        w = cp.Variable(self.n_assets)
        
        # Objective: minimize portfolio variance
        portfolio_variance = cp.quad_form(w, self.cov_matrix.values)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # weights sum to 1
            w >= min_weight,  # minimum weight
            w <= max_weight   # maximum weight
        ]
        
        # Add target return constraint if specified
        if target_return is not None:
            expected_return = self.mean_returns.values @ w
            constraints.append(expected_return >= target_return)
        
        # Solve optimization problem
        problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            weights = w.value
            portfolio_return = np.dot(weights, self.mean_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
            
            return {
                'weights': pd.Series(weights, index=self.asset_names),
                'expected_return': portfolio_return,
                'volatility': portfolio_risk,
                'sharpe_ratio': sharpe_ratio,
                'status': 'optimal'
            }
        else:
            return {'status': 'failed', 'message': problem.status}
    
    def maximum_sharpe_optimization(self, min_weight=0.0, max_weight=1.0):
        """
        Maximize Sharpe ratio optimization.
        
        Args:
            min_weight (float): Minimum weight constraint
            max_weight (float): Maximum weight constraint
            
        Returns:
            dict: Optimization results
        """
        # Define variables
        w = cp.Variable(self.n_assets)
        
        # Excess returns
        excess_returns = self.mean_returns - self.risk_free_rate
        
        # Objective: maximize Sharpe ratio (equivalent to minimizing negative Sharpe)
        portfolio_return = excess_returns.values @ w
        portfolio_variance = cp.quad_form(w, self.cov_matrix.values)
        
        # We use the inverse optimization trick: maximize return subject to variance <= 1
        constraints = [
            cp.sum(w) == 1,
            w >= min_weight,
            w <= max_weight,
            portfolio_variance <= 1
        ]
        
        problem = cp.Problem(cp.Maximize(portfolio_return), constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            weights = w.value
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            portfolio_return = np.dot(weights, self.mean_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
            
            return {
                'weights': pd.Series(weights, index=self.asset_names),
                'expected_return': portfolio_return,
                'volatility': portfolio_risk,
                'sharpe_ratio': sharpe_ratio,
                'status': 'optimal'
            }
        else:
            return {'status': 'failed', 'message': problem.status}
    
    def minimum_variance_optimization(self, min_weight=0.0, max_weight=1.0):
        """
        Minimum variance portfolio optimization.
        
        Args:
            min_weight (float): Minimum weight constraint
            max_weight (float): Maximum weight constraint
            
        Returns:
            dict: Optimization results
        """
        return self.markowitz_optimization(target_return=None, 
                                         min_weight=min_weight, 
                                         max_weight=max_weight)
    
    def risk_parity_optimization(self, target_risk=None):
        """
        Risk parity optimization where each asset contributes equally to portfolio risk.
        
        Args:
            target_risk (float, optional): Target portfolio volatility
            
        Returns:
            dict: Optimization results
        """
        def risk_parity_objective(weights):
            """Risk parity objective function."""
            portfolio_var = np.dot(weights, np.dot(self.cov_matrix, weights))
            marginal_contrib = np.dot(self.cov_matrix, weights)
            contrib = weights * marginal_contrib / portfolio_var
            
            # Minimize variance of risk contributions
            return np.sum((contrib - contrib.mean())**2)
        
        def portfolio_variance(weights):
            """Portfolio variance constraint."""
            return np.dot(weights, np.dot(self.cov_matrix, weights))
        
        # Initial guess: equal weights
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        ]
        
        if target_risk is not None:
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: portfolio_variance(x) - target_risk**2
            })
        
        # Bounds: all weights between 0 and 1
        bounds = [(0, 1) for _ in range(self.n_assets)]
        
        # Solve optimization
        result = minimize(risk_parity_objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            portfolio_return = np.dot(weights, self.mean_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
            
            return {
                'weights': pd.Series(weights, index=self.asset_names),
                'expected_return': portfolio_return,
                'volatility': portfolio_risk,
                'sharpe_ratio': sharpe_ratio,
                'status': 'optimal'
            }
        else:
            return {'status': 'failed', 'message': result.message}
    
    def black_litterman_optimization(self, views, confidence_matrix, tau=0.025):
        """
        Black-Litterman portfolio optimization incorporating investor views.
        
        Args:
            views (pd.Series): Investor views on expected returns
            confidence_matrix (np.array): Confidence in views (Omega matrix)
            tau (float): Uncertainty parameter
            
        Returns:
            dict: Optimization results with Black-Litterman adjusted returns
        """
        # Market capitalization weights (simplified: equal weights as proxy)
        market_weights = np.ones(self.n_assets) / self.n_assets
        
        # Implied equilibrium returns
        risk_aversion = 3.0  # Typical value
        pi = risk_aversion * np.dot(self.cov_matrix, market_weights)
        
        # Black-Litterman formula
        tau_cov = tau * self.cov_matrix.values
        
        # Create picking matrix P (identity for absolute views)
        P = np.eye(self.n_assets)
        
        # Black-Litterman adjusted expected returns
        M1 = np.linalg.inv(tau_cov)
        M2 = np.dot(P.T, np.dot(np.linalg.inv(confidence_matrix), P))
        M3 = np.dot(np.linalg.inv(tau_cov), pi)
        M4 = np.dot(P.T, np.dot(np.linalg.inv(confidence_matrix), views.values))
        
        mu_bl = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
        cov_bl = np.linalg.inv(M1 + M2)
        
        # Optimize with Black-Litterman inputs
        w = cp.Variable(self.n_assets)
        portfolio_variance = cp.quad_form(w, cov_bl)
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= 1
        ]
        
        problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            weights = w.value
            portfolio_return = np.dot(weights, mu_bl)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_bl, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
            
            return {
                'weights': pd.Series(weights, index=self.asset_names),
                'expected_return': portfolio_return,
                'volatility': portfolio_risk,
                'sharpe_ratio': sharpe_ratio,
                'bl_returns': pd.Series(mu_bl, index=self.asset_names),
                'status': 'optimal'
            }
        else:
            return {'status': 'failed', 'message': problem.status}
    
    def mean_cvar_optimization(self, alpha=0.05, target_return=None):
        """
        Mean-CVaR optimization (Conditional Value at Risk).
        
        Args:
            alpha (float): Confidence level for CVaR (e.g., 0.05 for 95% CVaR)
            target_return (float, optional): Target portfolio return
            
        Returns:
            dict: Optimization results
        """
        n_scenarios = len(self.returns)
        
        # Variables
        w = cp.Variable(self.n_assets)  # portfolio weights
        z = cp.Variable(n_scenarios)    # auxiliary variables for CVaR
        gamma = cp.Variable()           # VaR variable
        
        # Portfolio returns for each scenario
        portfolio_returns = self.returns.values @ w
        
        # CVaR constraints
        constraints = [
            cp.sum(w) == 1,  # weights sum to 1
            w >= 0,          # long-only
            w <= 1,          # maximum weight
            z >= 0,          # auxiliary variables non-negative
            z >= -portfolio_returns - gamma  # CVaR constraints
        ]
        
        # Target return constraint
        if target_return is not None:
            expected_return = self.mean_returns.values @ w
            constraints.append(expected_return >= target_return)
        
        # Objective: minimize CVaR
        cvar = gamma + (1/alpha) * cp.sum(z) / n_scenarios
        
        problem = cp.Problem(cp.Minimize(cvar), constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            weights = w.value
            portfolio_return = np.dot(weights, self.mean_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
            
            # Calculate VaR and CVaR
            portfolio_returns_realized = np.dot(self.returns.values, weights)
            var = np.percentile(portfolio_returns_realized, alpha * 100)
            cvar_value = np.mean(portfolio_returns_realized[portfolio_returns_realized <= var])
            
            return {
                'weights': pd.Series(weights, index=self.asset_names),
                'expected_return': portfolio_return,
                'volatility': portfolio_risk,
                'var': var,
                'cvar': cvar_value,
                'status': 'optimal'
            }
        else:
            return {'status': 'failed', 'message': problem.status}
    
    def efficient_frontier(self, n_points=50, min_weight=0.0, max_weight=1.0):
        """
        Generate the efficient frontier.
        
        Args:
            n_points (int): Number of points on the frontier
            min_weight (float): Minimum weight constraint
            max_weight (float): Maximum weight constraint
            
        Returns:
            pd.DataFrame: Efficient frontier data
        """
        # Calculate range of returns
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier_data = []
        
        for target_ret in target_returns:
            result = self.markowitz_optimization(
                target_return=target_ret,
                min_weight=min_weight,
                max_weight=max_weight
            )
            
            if result['status'] == 'optimal':
                frontier_data.append({
                    'return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio']
                })
        
        return pd.DataFrame(frontier_data)
    
    def get_portfolio_summary(self, weights):
        """
        Get comprehensive portfolio summary statistics.
        
        Args:
            weights (pd.Series or np.array): Portfolio weights
            
        Returns:
            dict: Portfolio statistics
        """
        if isinstance(weights, pd.Series):
            weights = weights.values
        
        # Basic metrics
        portfolio_return = np.dot(weights, self.mean_returns)
        portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Risk contributions
        marginal_contrib = np.dot(self.cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_variance
        
        # Diversification ratio
        weighted_vol = np.dot(weights, np.sqrt(np.diag(self.cov_matrix)))
        diversification_ratio = weighted_vol / portfolio_volatility
        
        # Effective number of assets (inverse Herfindahl index)
        effective_assets = 1 / np.sum(weights**2)
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'diversification_ratio': diversification_ratio,
            'effective_assets': effective_assets,
            'risk_contributions': pd.Series(risk_contrib, index=self.asset_names),
            'weights': pd.Series(weights, index=self.asset_names)
        }
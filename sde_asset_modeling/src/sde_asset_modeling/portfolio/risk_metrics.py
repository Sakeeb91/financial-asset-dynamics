"""
Comprehensive Risk Metrics and Value at Risk Calculations

This module implements various risk measures including:
- Value at Risk (VaR) - Historical, Parametric, Monte Carlo
- Conditional Value at Risk (CVaR/Expected Shortfall)
- Maximum Drawdown and Drawdown Duration
- Beta, Tracking Error, Information Ratio
- Risk-adjusted performance metrics
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')


class RiskMetrics:
    """
    Comprehensive risk metrics calculator for portfolios and individual assets.
    """
    
    def __init__(self, returns_data, benchmark_returns=None, risk_free_rate=0.02):
        """
        Initialize risk metrics calculator.
        
        Args:
            returns_data (pd.Series or pd.DataFrame): Asset/portfolio returns
            benchmark_returns (pd.Series, optional): Benchmark returns for relative metrics
            risk_free_rate (float): Risk-free rate for excess return calculations
        """
        self.returns = returns_data
        self.benchmark = benchmark_returns
        self.risk_free_rate = risk_free_rate
        
        # Ensure returns are in DataFrame format
        if isinstance(returns_data, pd.Series):
            self.returns = returns_data.to_frame('returns')
    
    def value_at_risk(self, confidence_level=0.05, method='historical'):
        """
        Calculate Value at Risk (VaR) using different methods.
        
        Args:
            confidence_level (float): Confidence level (e.g., 0.05 for 95% VaR)
            method (str): 'historical', 'parametric', or 'monte_carlo'
            
        Returns:
            dict: VaR estimates for each method/asset
        """
        results = {}
        
        for col in self.returns.columns:
            returns = self.returns[col].dropna()
            
            if method == 'historical':
                var = np.percentile(returns, confidence_level * 100)
            
            elif method == 'parametric':
                mu = returns.mean()
                sigma = returns.std()
                var = stats.norm.ppf(confidence_level, mu, sigma)
            
            elif method == 'monte_carlo':
                # Fit normal distribution and simulate
                mu, sigma = returns.mean(), returns.std()
                simulated_returns = np.random.normal(mu, sigma, 10000)
                var = np.percentile(simulated_returns, confidence_level * 100)
            
            else:
                raise ValueError("Method must be 'historical', 'parametric', or 'monte_carlo'")
            
            results[col] = var
        
        return results
    
    def conditional_value_at_risk(self, confidence_level=0.05, method='historical'):
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            confidence_level (float): Confidence level
            method (str): 'historical', 'parametric', or 'monte_carlo'
            
        Returns:
            dict: CVaR estimates
        """
        results = {}
        
        for col in self.returns.columns:
            returns = self.returns[col].dropna()
            
            if method == 'historical':
                var = np.percentile(returns, confidence_level * 100)
                cvar = returns[returns <= var].mean()
            
            elif method == 'parametric':
                mu, sigma = returns.mean(), returns.std()
                var = stats.norm.ppf(confidence_level, mu, sigma)
                # Analytical CVaR for normal distribution
                cvar = mu - sigma * stats.norm.pdf(stats.norm.ppf(confidence_level)) / confidence_level
            
            elif method == 'monte_carlo':
                mu, sigma = returns.mean(), returns.std()
                simulated_returns = np.random.normal(mu, sigma, 10000)
                var = np.percentile(simulated_returns, confidence_level * 100)
                cvar = simulated_returns[simulated_returns <= var].mean()
            
            results[col] = cvar
        
        return results
    
    def maximum_drawdown(self):
        """
        Calculate maximum drawdown and related metrics.
        
        Returns:
            dict: Maximum drawdown statistics
        """
        results = {}
        
        for col in self.returns.columns:
            returns = self.returns[col].dropna()
            
            # Calculate cumulative returns
            cumulative = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cumulative.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max
            
            # Maximum drawdown
            max_dd = drawdown.min()
            
            # Maximum drawdown duration
            dd_duration = 0
            current_duration = 0
            
            for dd in drawdown:
                if dd < 0:
                    current_duration += 1
                    dd_duration = max(dd_duration, current_duration)
                else:
                    current_duration = 0
            
            # Recovery time (time to recover from max drawdown)
            max_dd_end = drawdown.idxmin()
            try:
                recovery_start = drawdown[max_dd_end:][drawdown[max_dd_end:] >= 0].index[0]
                recovery_time = len(drawdown[max_dd_end:recovery_start])
            except IndexError:
                recovery_time = np.nan  # Still in drawdown
            
            results[col] = {
                'max_drawdown': max_dd,
                'max_drawdown_duration': dd_duration,
                'recovery_time': recovery_time,
                'current_drawdown': drawdown.iloc[-1],
                'drawdown_series': drawdown
            }
        
        return results
    
    def volatility_metrics(self, window=None):
        """
        Calculate various volatility metrics.
        
        Args:
            window (int, optional): Rolling window for calculations
            
        Returns:
            dict: Volatility statistics
        """
        results = {}
        
        for col in self.returns.columns:
            returns = self.returns[col].dropna()
            
            # Annualized volatility (assuming daily returns)
            annual_vol = returns.std() * np.sqrt(252)
            
            # Rolling volatility if window specified
            if window:
                rolling_vol = returns.rolling(window).std() * np.sqrt(252)
            else:
                rolling_vol = None
            
            # Downside deviation (semi-volatility)
            downside_returns = returns[returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # Upside volatility
            upside_returns = returns[returns > 0]
            upside_vol = upside_returns.std() * np.sqrt(252) if len(upside_returns) > 0 else 0
            
            # GARCH volatility estimation
            try:
                garch_model = arch_model(returns * 100, vol='Garch', p=1, q=1)
                garch_fit = garch_model.fit(disp='off')
                garch_vol = garch_fit.conditional_volatility.iloc[-1] / 100 * np.sqrt(252)
            except:
                garch_vol = np.nan
            
            results[col] = {
                'annual_volatility': annual_vol,
                'rolling_volatility': rolling_vol,
                'downside_volatility': downside_vol,
                'upside_volatility': upside_vol,
                'garch_volatility': garch_vol,
                'volatility_ratio': upside_vol / downside_vol if downside_vol > 0 else np.inf
            }
        
        return results
    
    def performance_metrics(self):
        """
        Calculate risk-adjusted performance metrics.
        
        Returns:
            dict: Performance statistics
        """
        results = {}
        
        for col in self.returns.columns:
            returns = self.returns[col].dropna()
            
            # Basic metrics
            annual_return = returns.mean() * 252
            annual_vol = returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            sharpe = (annual_return - self.risk_free_rate) / annual_vol
            
            # Sortino ratio (using downside deviation)
            downside_returns = returns[returns < self.risk_free_rate/252]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annual_vol
            sortino = (annual_return - self.risk_free_rate) / downside_vol
            
            # Calmar ratio (return / max drawdown)
            max_dd = self.maximum_drawdown()[col]['max_drawdown']
            calmar = annual_return / abs(max_dd) if max_dd != 0 else np.inf
            
            # Omega ratio
            threshold = self.risk_free_rate / 252
            gains = returns[returns > threshold] - threshold
            losses = threshold - returns[returns <= threshold]
            omega = gains.sum() / losses.sum() if losses.sum() > 0 else np.inf
            
            # Information ratio (if benchmark provided)
            if self.benchmark is not None:
                tracking_error = (returns - self.benchmark).std() * np.sqrt(252)
                excess_return = annual_return - (self.benchmark.mean() * 252)
                info_ratio = excess_return / tracking_error if tracking_error > 0 else np.nan
            else:
                info_ratio = np.nan
                tracking_error = np.nan
            
            # Beta (if benchmark provided)
            if self.benchmark is not None:
                covariance = np.cov(returns, self.benchmark)[0, 1]
                benchmark_var = np.var(self.benchmark)
                beta = covariance / benchmark_var if benchmark_var > 0 else np.nan
            else:
                beta = np.nan
            
            # Treynor ratio
            treynor = (annual_return - self.risk_free_rate) / beta if not np.isnan(beta) and beta != 0 else np.nan
            
            # Jensen's alpha
            if not np.isnan(beta):
                expected_return = self.risk_free_rate + beta * (self.benchmark.mean() * 252 - self.risk_free_rate)
                jensen_alpha = annual_return - expected_return
            else:
                jensen_alpha = np.nan
            
            results[col] = {
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'omega_ratio': omega,
                'information_ratio': info_ratio,
                'tracking_error': tracking_error,
                'beta': beta,
                'treynor_ratio': treynor,
                'jensen_alpha': jensen_alpha
            }
        
        return results
    
    def tail_risk_metrics(self):
        """
        Calculate tail risk measures.
        
        Returns:
            dict: Tail risk statistics
        """
        results = {}
        
        for col in self.returns.columns:
            returns = self.returns[col].dropna()
            
            # Skewness and kurtosis
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            excess_kurtosis = kurtosis - 3
            
            # Jarque-Bera test for normality
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
            
            # Left tail index (Hill estimator)
            negative_returns = -returns[returns < 0].sort_values()
            if len(negative_returns) > 10:
                k = min(50, len(negative_returns) // 4)  # Number of order statistics
                tail_index = 1 / np.mean(np.log(negative_returns[:k]) - np.log(negative_returns[k]))
            else:
                tail_index = np.nan
            
            # Extreme value statistics
            extreme_losses = returns[returns < np.percentile(returns, 5)]
            extreme_gains = returns[returns > np.percentile(returns, 95)]
            
            results[col] = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'excess_kurtosis': excess_kurtosis,
                'jb_statistic': jb_stat,
                'jb_pvalue': jb_pvalue,
                'is_normal': jb_pvalue > 0.05,
                'tail_index': tail_index,
                'extreme_loss_mean': extreme_losses.mean(),
                'extreme_gain_mean': extreme_gains.mean(),
                'extreme_ratio': abs(extreme_gains.mean() / extreme_losses.mean()) if extreme_losses.mean() != 0 else np.inf
            }
        
        return results
    
    def risk_decomposition(self, portfolio_weights=None):
        """
        Decompose portfolio risk into component contributions.
        
        Args:
            portfolio_weights (dict or pd.Series, optional): Portfolio weights
            
        Returns:
            dict: Risk decomposition analysis
        """
        if portfolio_weights is None and len(self.returns.columns) == 1:
            return "Risk decomposition requires multiple assets or portfolio weights"
        
        if portfolio_weights is None:
            # Equal weights
            portfolio_weights = pd.Series(1/len(self.returns.columns), 
                                        index=self.returns.columns)
        
        if isinstance(portfolio_weights, dict):
            portfolio_weights = pd.Series(portfolio_weights)
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns * portfolio_weights).sum(axis=1)
        portfolio_var = portfolio_returns.var()
        
        # Individual asset contributions to portfolio variance
        cov_matrix = self.returns.cov()
        
        # Marginal contribution to risk (MCTR)
        mctr = {}
        for asset in self.returns.columns:
            marginal_contrib = 2 * sum(portfolio_weights[other] * cov_matrix.loc[asset, other] 
                                     for other in self.returns.columns)
            mctr[asset] = marginal_contrib / (2 * np.sqrt(portfolio_var))
        
        # Component contribution to risk (CCTR)
        cctr = {asset: portfolio_weights[asset] * mctr[asset] for asset in mctr}
        
        # Percentage contribution
        total_cctr = sum(cctr.values())
        pct_contrib = {asset: cctr[asset] / total_cctr * 100 for asset in cctr}
        
        return {
            'portfolio_volatility': np.sqrt(portfolio_var) * np.sqrt(252),
            'marginal_contributions': mctr,
            'component_contributions': cctr,
            'percentage_contributions': pct_contrib,
            'portfolio_weights': portfolio_weights.to_dict()
        }
    
    def stress_testing(self, stress_scenarios):
        """
        Perform stress testing on portfolio returns.
        
        Args:
            stress_scenarios (dict): Dictionary of stress scenarios
                e.g., {'market_crash': {'factor': -0.2}, 'volatility_spike': {'vol_mult': 2}}
            
        Returns:
            dict: Stress test results
        """
        results = {}
        
        for scenario_name, scenario_params in stress_scenarios.items():
            scenario_results = {}
            
            for col in self.returns.columns:
                returns = self.returns[col].dropna()
                
                if 'factor' in scenario_params:
                    # Apply multiplicative factor
                    stressed_returns = returns * (1 + scenario_params['factor'])
                elif 'shift' in scenario_params:
                    # Apply additive shift
                    stressed_returns = returns + scenario_params['shift']
                elif 'vol_mult' in scenario_params:
                    # Scale volatility
                    mean_ret = returns.mean()
                    vol_mult = scenario_params['vol_mult']
                    stressed_returns = mean_ret + (returns - mean_ret) * vol_mult
                else:
                    stressed_returns = returns
                
                # Calculate metrics for stressed scenario
                var_95 = np.percentile(stressed_returns, 5)
                cvar_95 = stressed_returns[stressed_returns <= var_95].mean()
                
                scenario_results[col] = {
                    'var_95': var_95,
                    'cvar_95': cvar_95,
                    'mean_return': stressed_returns.mean(),
                    'volatility': stressed_returns.std() * np.sqrt(252)
                }
            
            results[scenario_name] = scenario_results
        
        return results
    
    def generate_risk_report(self):
        """
        Generate a comprehensive risk report.
        
        Returns:
            dict: Complete risk analysis
        """
        report = {
            'var_estimates': {
                'historical_95': self.value_at_risk(0.05, 'historical'),
                'parametric_95': self.value_at_risk(0.05, 'parametric'),
                'historical_99': self.value_at_risk(0.01, 'historical')
            },
            'cvar_estimates': {
                'historical_95': self.conditional_value_at_risk(0.05, 'historical'),
                'parametric_95': self.conditional_value_at_risk(0.05, 'parametric')
            },
            'drawdown_metrics': self.maximum_drawdown(),
            'volatility_metrics': self.volatility_metrics(),
            'performance_metrics': self.performance_metrics(),
            'tail_risk_metrics': self.tail_risk_metrics()
        }
        
        # Add stress test results
        default_stress_scenarios = {
            'market_crash': {'factor': -0.3},
            'volatility_spike': {'vol_mult': 2},
            'interest_rate_shock': {'shift': 0.02}
        }
        report['stress_tests'] = self.stress_testing(default_stress_scenarios)
        
        return report
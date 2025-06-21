"""
Advanced Financial Analytics Dashboard

This creates a comprehensive dashboard showcasing:
- Portfolio optimization results
- Risk analytics and VaR calculations
- Options pricing and Greeks
- Market regime detection
- Interactive visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import sys
import os
sys.path.append('src')

from sde_asset_modeling.portfolio.optimizer import PortfolioOptimizer
from sde_asset_modeling.portfolio.risk_metrics import RiskMetrics
from sde_asset_modeling.options.black_scholes import BlackScholesModel
from sde_asset_modeling.analytics.regime_detection import RegimeDetector
from sde_asset_modeling.models.gbm import GBMModel
from sde_asset_modeling.models.heston import HestonModel
from sde_asset_modeling.simulation.engine import simulate_paths

def generate_synthetic_market_data():
    """Generate synthetic market data for demonstration."""
    np.random.seed(42)
    
    # Create 5 synthetic assets with different characteristics
    n_days = 1000
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # Asset parameters
    assets = {
        'Tech_Stock': {'mu': 0.12, 'sigma': 0.25, 'initial': 100},
        'Financial': {'mu': 0.08, 'sigma': 0.20, 'initial': 80},
        'Energy': {'mu': 0.06, 'sigma': 0.30, 'initial': 60},
        'Healthcare': {'mu': 0.10, 'sigma': 0.18, 'initial': 120},
        'REIT': {'mu': 0.07, 'sigma': 0.22, 'initial': 40}
    }
    
    # Generate correlated returns
    correlation_matrix = np.array([
        [1.0, 0.6, 0.3, 0.4, 0.2],
        [0.6, 1.0, 0.4, 0.5, 0.3],
        [0.3, 0.4, 1.0, 0.2, 0.1],
        [0.4, 0.5, 0.2, 1.0, 0.3],
        [0.2, 0.3, 0.1, 0.3, 1.0]
    ])
    
    # Generate random returns
    random_returns = np.random.multivariate_normal(
        mean=[0] * 5,
        cov=correlation_matrix,
        size=n_days
    )
    
    # Create price series
    prices_data = {}
    returns_data = {}
    
    for i, (name, params) in enumerate(assets.items()):
        # Scale returns by volatility and add drift
        daily_returns = params['mu']/252 + params['sigma']/np.sqrt(252) * random_returns[:, i]
        
        # Generate price series
        prices = [params['initial']]
        for ret in daily_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        prices_data[name] = prices
        returns_data[name] = daily_returns
    
    prices_df = pd.DataFrame(prices_data, index=dates)
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    return prices_df, returns_df

def create_portfolio_optimization_viz(returns_df):
    """Create portfolio optimization visualizations."""
    print("Creating portfolio optimization analysis...")
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(returns_df)
    
    # Generate efficient frontier
    frontier = optimizer.efficient_frontier(n_points=50)
    
    # Optimize for different objectives
    max_sharpe = optimizer.maximum_sharpe_optimization()
    min_var = optimizer.minimum_variance_optimization()
    risk_parity = optimizer.risk_parity_optimization()
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Efficient Frontier',
            'Portfolio Weights Comparison',
            'Risk Contributions',
            'Performance Metrics'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Efficient frontier
    fig.add_trace(
        go.Scatter(
            x=frontier['volatility'],
            y=frontier['return'],
            mode='lines+markers',
            name='Efficient Frontier',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add optimal portfolios
    portfolios = {
        'Max Sharpe': max_sharpe,
        'Min Variance': min_var,
        'Risk Parity': risk_parity
    }
    
    colors = ['red', 'green', 'orange']
    for i, (name, portfolio) in enumerate(portfolios.items()):
        if portfolio['status'] == 'optimal':
            fig.add_trace(
                go.Scatter(
                    x=[portfolio['volatility']],
                    y=[portfolio['expected_return']],
                    mode='markers',
                    name=name,
                    marker=dict(size=12, color=colors[i])
                ),
                row=1, col=1
            )
    
    # Portfolio weights comparison
    weights_df = pd.DataFrame({
        name: portfolio['weights'] if portfolio['status'] == 'optimal' else pd.Series(0, index=returns_df.columns)
        for name, portfolio in portfolios.items()
    })
    
    for i, portfolio_name in enumerate(weights_df.columns):
        fig.add_trace(
            go.Bar(
                x=weights_df.index,
                y=weights_df[portfolio_name],
                name=f'{portfolio_name} Weights',
                marker_color=colors[i]
            ),
            row=1, col=2
        )
    
    # Risk contributions for Risk Parity portfolio
    if risk_parity['status'] == 'optimal':
        summary = optimizer.get_portfolio_summary(risk_parity['weights'])
        fig.add_trace(
            go.Bar(
                x=summary['risk_contributions'].index,
                y=summary['risk_contributions'].values,
                name='Risk Contributions',
                marker_color='purple'
            ),
            row=2, col=1
        )
    
    # Performance metrics comparison
    metrics_data = []
    for name, portfolio in portfolios.items():
        if portfolio['status'] == 'optimal':
            metrics_data.append({
                'Portfolio': name,
                'Expected Return': portfolio['expected_return'],
                'Volatility': portfolio['volatility'],
                'Sharpe Ratio': portfolio['sharpe_ratio']
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    if not metrics_df.empty:
        fig.add_trace(
            go.Bar(
                x=metrics_df['Portfolio'],
                y=metrics_df['Sharpe Ratio'],
                name='Sharpe Ratio',
                marker_color='darkblue'
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='Portfolio Optimization Analysis',
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(title_text='Volatility', row=1, col=1)
    fig.update_yaxes(title_text='Expected Return', row=1, col=1)
    fig.update_xaxes(title_text='Assets', row=1, col=2)
    fig.update_yaxes(title_text='Weight', row=1, col=2)
    fig.update_xaxes(title_text='Assets', row=2, col=1)
    fig.update_yaxes(title_text='Risk Contribution', row=2, col=1)
    fig.update_xaxes(title_text='Portfolio', row=2, col=2)
    fig.update_yaxes(title_text='Sharpe Ratio', row=2, col=2)
    
    return fig, portfolios

def create_risk_analytics_viz(returns_df):
    """Create comprehensive risk analytics dashboard."""
    print("Creating risk analytics...")
    
    # Calculate risk metrics for each asset
    risk_metrics = RiskMetrics(returns_df)
    
    # Generate risk report
    risk_report = risk_metrics.generate_risk_report()
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Value at Risk (95%)',
            'Maximum Drawdown',
            'Performance Metrics',
            'Tail Risk Analysis'
        ]
    )
    
    # VaR comparison
    var_data = risk_report['var_estimates']['historical_95']
    fig.add_trace(
        go.Bar(
            x=list(var_data.keys()),
            y=list(var_data.values()),
            name='VaR 95%',
            marker_color='red'
        ),
        row=1, col=1
    )
    
    # Maximum drawdown
    dd_data = risk_report['drawdown_metrics']
    max_dd_values = [dd_data[asset]['max_drawdown'] for asset in dd_data.keys()]
    fig.add_trace(
        go.Bar(
            x=list(dd_data.keys()),
            y=max_dd_values,
            name='Max Drawdown',
            marker_color='darkred'
        ),
        row=1, col=2
    )
    
    # Performance metrics
    perf_data = risk_report['performance_metrics']
    sharpe_ratios = [perf_data[asset]['sharpe_ratio'] for asset in perf_data.keys()]
    fig.add_trace(
        go.Bar(
            x=list(perf_data.keys()),
            y=sharpe_ratios,
            name='Sharpe Ratio',
            marker_color='green'
        ),
        row=2, col=1
    )
    
    # Tail risk metrics
    tail_data = risk_report['tail_risk_metrics']
    skewness_values = [tail_data[asset]['skewness'] for asset in tail_data.keys()]
    fig.add_trace(
        go.Bar(
            x=list(tail_data.keys()),
            y=skewness_values,
            name='Skewness',
            marker_color='orange'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Risk Analytics Dashboard',
        height=800,
        showlegend=False
    )
    
    return fig, risk_report

def create_options_analytics_viz():
    """Create options pricing and Greeks visualization."""
    print("Creating options analytics...")
    
    # Create Black-Scholes model
    bs_model = BlackScholesModel(
        spot_price=100,
        strike_price=100,
        time_to_expiry=0.25,  # 3 months
        risk_free_rate=0.05,
        volatility=0.2
    )
    
    # Generate option chain
    strikes = np.arange(80, 121, 2.5)
    option_chain = bs_model.option_chain(strikes, option_type='both')
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Option Prices vs Strike',
            'Greeks - Delta',
            'Greeks - Gamma',
            'Greeks - Vega'
        ]
    )
    
    # Option prices
    fig.add_trace(
        go.Scatter(
            x=option_chain['strike'],
            y=option_chain['call_price'],
            mode='lines',
            name='Call Price',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=option_chain['strike'],
            y=option_chain['put_price'],
            mode='lines',
            name='Put Price',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # Delta
    fig.add_trace(
        go.Scatter(
            x=option_chain['strike'],
            y=option_chain['call_delta'],
            mode='lines',
            name='Call Delta',
            line=dict(color='blue')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=option_chain['strike'],
            y=option_chain['put_delta'],
            mode='lines',
            name='Put Delta',
            line=dict(color='red')
        ),
        row=1, col=2
    )
    
    # Gamma
    fig.add_trace(
        go.Scatter(
            x=option_chain['strike'],
            y=option_chain['call_gamma'],
            mode='lines',
            name='Gamma',
            line=dict(color='green')
        ),
        row=2, col=1
    )
    
    # Vega
    fig.add_trace(
        go.Scatter(
            x=option_chain['strike'],
            y=option_chain['call_vega'],
            mode='lines',
            name='Vega',
            line=dict(color='purple')
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Options Analytics Dashboard',
        height=800
    )
    
    return fig, option_chain

def create_regime_detection_viz(returns_df):
    """Create market regime detection visualization."""
    print("Creating regime analysis...")
    
    # Use first asset for regime detection
    asset_returns = returns_df.iloc[:, 0]
    regime_detector = RegimeDetector(asset_returns)
    
    # Generate regime analysis
    regime_report = regime_detector.generate_regime_report()
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Volatility Regimes',
            'Regime Probabilities',
            'Structural Breaks',
            'Regime Statistics'
        ]
    )
    
    # Volatility clustering regimes
    if 'volatility_clustering' in regime_report and 'error' not in regime_report['volatility_clustering']:
        vol_regimes = regime_report['volatility_clustering']['regimes']
        regime_colors = ['green', 'yellow', 'red']
        
        for regime in [0, 1, 2]:
            regime_mask = vol_regimes == regime
            regime_dates = vol_regimes[regime_mask].index
            regime_returns = asset_returns[regime_mask]
            
            if len(regime_returns) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=regime_dates,
                        y=regime_returns.cumsum(),
                        mode='markers',
                        name=f'Regime {regime}',
                        marker=dict(color=regime_colors[regime], size=3)
                    ),
                    row=1, col=1
                )
    
    # Gaussian mixture regime probabilities
    if 'gaussian_mixture' in regime_report and 'error' not in regime_report['gaussian_mixture']:
        gmm_regimes = regime_report['gaussian_mixture']['regimes']
        
        for i in range(3):  # Assuming 3 regimes
            col_name = f'prob_regime_{i}'
            if col_name in gmm_regimes.columns:
                fig.add_trace(
                    go.Scatter(
                        x=gmm_regimes.index,
                        y=gmm_regimes[col_name],
                        mode='lines',
                        name=f'Regime {i} Prob',
                        line=dict(width=1)
                    ),
                    row=1, col=2
                )
    
    # Structural breaks
    if 'structural_breaks' in regime_report and 'error' not in regime_report['structural_breaks']:
        cusum_stats = regime_report['structural_breaks']['cusum_statistics']
        fig.add_trace(
            go.Scatter(
                x=cusum_stats.index,
                y=cusum_stats.values,
                mode='lines',
                name='CUSUM Statistics',
                line=dict(color='black')
            ),
            row=2, col=1
        )
        
        # Mark break points
        break_points = regime_report['structural_breaks']['break_points']
        if len(break_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=break_points,
                    y=[cusum_stats.loc[bp] for bp in break_points],
                    mode='markers',
                    name='Break Points',
                    marker=dict(color='red', size=10, symbol='x')
                ),
                row=2, col=1
            )
    
    # Regime statistics
    if 'volatility_clustering' in regime_report and 'error' not in regime_report['volatility_clustering']:
        regime_stats = regime_report['volatility_clustering']['regime_stats']
        regime_names = []
        regime_returns = []
        regime_vols = []
        
        for regime, stats in regime_stats.items():
            regime_names.append(stats['name'])
            regime_returns.append(stats['mean_return'])
            regime_vols.append(stats['volatility'])
        
        fig.add_trace(
            go.Scatter(
                x=regime_vols,
                y=regime_returns,
                mode='markers+text',
                text=regime_names,
                textposition='top center',
                name='Regime Stats',
                marker=dict(size=15, color=['green', 'yellow', 'red'])
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title='Market Regime Analysis',
        height=800
    )
    
    return fig, regime_report

def create_simulation_comparison_viz():
    """Create SDE simulation comparison visualization."""
    print("Creating SDE simulation comparison...")
    
    # Simulation parameters
    t_span = (0.0, 1.0)
    dt = 1/252
    n_paths = 1000
    
    # GBM simulation
    gbm_model = GBMModel(mu=0.1, sigma=0.2, x0=100)
    t_gbm, paths_gbm = simulate_paths(gbm_model, t_span, dt, n_paths)
    
    # Heston simulation (simplified)
    heston_model = HestonModel(
        mu=0.1, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, s0=100, v0=0.04
    )
    
    # For demonstration, we'll create a simple simulation
    np.random.seed(42)
    t_heston = t_gbm
    paths_heston = []
    
    for _ in range(min(100, n_paths)):  # Limit for demo
        S = [100]
        v = [0.04]
        
        for i in range(1, len(t_heston)):
            dt_sim = t_heston[i] - t_heston[i-1]
            
            # Simple Euler scheme for Heston
            dW1 = np.random.normal(0, np.sqrt(dt_sim))
            dW2 = np.random.normal(0, np.sqrt(dt_sim))
            
            # Correlated Brownian motions
            dW2_corr = heston_model.rho * dW1 + np.sqrt(1 - heston_model.rho**2) * dW2
            
            # Update variance (with floor to prevent negative values)
            dv = heston_model.kappa * (heston_model.theta - v[-1]) * dt_sim + heston_model.xi * np.sqrt(max(v[-1], 0)) * dW2_corr
            v_new = max(v[-1] + dv, 0.001)
            v.append(v_new)
            
            # Update price
            dS = heston_model.mu * S[-1] * dt_sim + np.sqrt(v[-1]) * S[-1] * dW1
            S.append(S[-1] + dS)
        
        paths_heston.append(S)
    
    paths_heston = np.array(paths_heston)
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'GBM Sample Paths',
            'Heston Sample Paths',
            'Return Distributions',
            'Volatility Clustering (Heston)'
        ]
    )
    
    # GBM paths (sample)
    for i in range(min(20, paths_gbm.shape[0])):
        fig.add_trace(
            go.Scatter(
                x=t_gbm,
                y=paths_gbm[i],
                mode='lines',
                line=dict(width=1, color='blue'),
                opacity=0.3,
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Heston paths (sample)
    for i in range(min(20, len(paths_heston))):
        fig.add_trace(
            go.Scatter(
                x=t_heston,
                y=paths_heston[i],
                mode='lines',
                line=dict(width=1, color='red'),
                opacity=0.3,
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Return distributions
    gbm_returns = np.diff(np.log(paths_gbm), axis=1).flatten()
    heston_returns = np.diff(np.log(paths_heston), axis=1).flatten()
    
    fig.add_trace(
        go.Histogram(
            x=gbm_returns,
            name='GBM Returns',
            opacity=0.7,
            nbinsx=50,
            histnorm='probability'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Histogram(
            x=heston_returns,
            name='Heston Returns',
            opacity=0.7,
            nbinsx=50,
            histnorm='probability'
        ),
        row=2, col=1
    )
    
    # Volatility clustering for Heston
    if len(paths_heston) > 0:
        sample_path = paths_heston[0]
        sample_returns = np.diff(np.log(sample_path))
        rolling_vol = pd.Series(sample_returns).rolling(window=21).std() * np.sqrt(252)
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(rolling_vol))),
                y=rolling_vol,
                mode='lines',
                name='Rolling Volatility',
                line=dict(color='purple')
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title='SDE Model Comparison',
        height=800
    )
    
    return fig

def main():
    """Create comprehensive advanced financial dashboard."""
    print("Generating Advanced Financial Analytics Dashboard...")
    print("=" * 50)
    
    # Generate synthetic market data
    prices_df, returns_df = generate_synthetic_market_data()
    print(f"Generated market data: {len(prices_df)} days, {len(prices_df.columns)} assets")
    
    # Create all visualizations
    portfolio_fig, portfolios = create_portfolio_optimization_viz(returns_df)
    risk_fig, risk_report = create_risk_analytics_viz(returns_df)
    options_fig, option_chain = create_options_analytics_viz()
    regime_fig, regime_report = create_regime_detection_viz(returns_df)
    simulation_fig = create_simulation_comparison_viz()
    
    # Save all figures
    print("\\nSaving visualizations...")
    
    portfolio_fig.write_html('advanced_portfolio_optimization.html')
    portfolio_fig.write_image('advanced_portfolio_optimization.png', width=1200, height=800)
    print("✓ Portfolio optimization dashboard saved")
    
    risk_fig.write_html('advanced_risk_analytics.html')
    risk_fig.write_image('advanced_risk_analytics.png', width=1200, height=800)
    print("✓ Risk analytics dashboard saved")
    
    options_fig.write_html('advanced_options_analytics.html')
    options_fig.write_image('advanced_options_analytics.png', width=1200, height=800)
    print("✓ Options analytics dashboard saved")
    
    regime_fig.write_html('advanced_regime_detection.html')
    regime_fig.write_image('advanced_regime_detection.png', width=1200, height=800)
    print("✓ Regime detection dashboard saved")
    
    simulation_fig.write_html('advanced_sde_comparison.html')
    simulation_fig.write_image('advanced_sde_comparison.png', width=1200, height=800)
    print("✓ SDE comparison dashboard saved")
    
    # Print summary statistics
    print("\\n" + "=" * 50)
    print("ADVANCED ANALYTICS SUMMARY")
    print("=" * 50)
    
    print("\\n📊 Portfolio Optimization Results:")
    for name, portfolio in portfolios.items():
        if portfolio['status'] == 'optimal':
            print(f"  {name}:")
            print(f"    Expected Return: {portfolio['expected_return']:.2%}")
            print(f"    Volatility: {portfolio['volatility']:.2%}")
            print(f"    Sharpe Ratio: {portfolio['sharpe_ratio']:.3f}")
    
    print("\\n⚠️  Risk Analytics:")
    if 'Tech_Stock' in risk_report['var_estimates']['historical_95']:
        var_95 = risk_report['var_estimates']['historical_95']['Tech_Stock']
        print(f"  Tech Stock VaR (95%): {var_95:.2%}")
    
    print("\\n🎯 Options Analytics:")
    atm_call = option_chain[option_chain['strike'] == 100]['call_price'].iloc[0]
    atm_put = option_chain[option_chain['strike'] == 100]['put_price'].iloc[0]
    print(f"  ATM Call Price: ${atm_call:.2f}")
    print(f"  ATM Put Price: ${atm_put:.2f}")
    
    print("\\n🔄 Market Regimes:")
    if 'volatility_clustering' in regime_report and 'error' not in regime_report['volatility_clustering']:
        regime_stats = regime_report['volatility_clustering']['regime_stats']
        for regime, stats in regime_stats.items():
            print(f"  {stats['name']}: {stats['frequency']:.1%} of time")
    
    print("\\n" + "=" * 50)
    print("✅ Advanced Financial Analytics Dashboard Complete!")
    print("📁 Check the generated HTML and PNG files for interactive visualizations")
    print("🚀 This demonstrates sophisticated quantitative finance knowledge!")

if __name__ == "__main__":
    main()
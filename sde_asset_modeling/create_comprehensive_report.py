"""
Comprehensive Financial Analysis Report Generator

This script creates a professional-grade financial analysis report that demonstrates
advanced quantitative finance capabilities including:
- Real market data analysis
- Portfolio optimization
- Risk management
- Options pricing
- Market regime detection
- Performance attribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
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
from sde_asset_modeling.analytics.market_data import MarketDataManager

class ComprehensiveReportGenerator:
    """
    Generate comprehensive financial analysis reports.
    """
    
    def __init__(self):
        self.market_data_manager = MarketDataManager()
        self.report_data = {}
        
    def generate_executive_summary(self, market_report, portfolio_results, risk_analysis):
        """Generate executive summary of the analysis."""
        current_date = datetime.now().strftime("%B %d, %Y")
        
        summary = {
            'report_date': current_date,
            'analysis_period': market_report['period'],
            'assets_analyzed': len(market_report['symbols']),
            'key_findings': []
        }
        
        # Market conditions assessment
        if 'current_conditions' in market_report:
            conditions = market_report['current_conditions']
            volatility = conditions.get('market_volatility', 0)
            correlation = conditions.get('average_correlation', 0)
            
            if volatility > 25:
                summary['key_findings'].append("High market volatility detected - elevated risk environment")
            elif volatility < 15:
                summary['key_findings'].append("Low volatility environment - potential complacency risk")
            
            if correlation > 0.7:
                summary['key_findings'].append("High asset correlations - reduced diversification benefits")
            elif correlation < 0.3:
                summary['key_findings'].append("Low correlations - good diversification opportunities")
        
        # Portfolio optimization insights
        if portfolio_results and 'Max Sharpe' in portfolio_results:
            max_sharpe = portfolio_results['Max Sharpe']
            if max_sharpe['status'] == 'optimal':
                sharpe = max_sharpe['sharpe_ratio']
                if sharpe > 1.5:
                    summary['key_findings'].append("Excellent risk-adjusted returns achievable")
                elif sharpe < 0.5:
                    summary['key_findings'].append("Limited attractive investment opportunities identified")
        
        # Risk assessment
        if risk_analysis:
            # This would be populated based on risk metrics
            summary['key_findings'].append("Comprehensive risk metrics calculated across all assets")
        
        return summary
    
    def create_market_overview_section(self, market_report):
        """Create market overview visualizations."""
        data = market_report['data']
        metrics = market_report['individual_metrics']
        
        # Price performance chart
        prices = data['prices']
        normalized_prices = prices / prices.iloc[0] * 100
        
        fig_performance = go.Figure()
        
        for symbol in normalized_prices.columns:
            fig_performance.add_trace(
                go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices[symbol],
                    mode='lines',
                    name=symbol,
                    line=dict(width=2)
                )
            )
        
        fig_performance.update_layout(
            title='Normalized Price Performance (Base = 100)',
            xaxis_title='Date',
            yaxis_title='Normalized Price',
            height=500,
            template='plotly_white'
        )
        
        # Risk-return scatter plot
        fig_risk_return = go.Figure()
        
        returns = [metrics[symbol]['annual_return'] for symbol in metrics.keys()]
        volatilities = [metrics[symbol]['annual_volatility'] for symbol in metrics.keys()]
        symbols = list(metrics.keys())
        
        fig_risk_return.add_trace(
            go.Scatter(
                x=volatilities,
                y=returns,
                mode='markers+text',
                text=symbols,
                textposition='top center',
                marker=dict(
                    size=12,
                    color=returns,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Annual Return")
                ),
                name='Assets'
            )
        )
        
        fig_risk_return.update_layout(
            title='Risk-Return Profile',
            xaxis_title='Annual Volatility',
            yaxis_title='Annual Return',
            height=500,
            template='plotly_white'
        )
        
        # Performance metrics table
        metrics_df = pd.DataFrame(metrics).T
        metrics_df = metrics_df.round(4)
        
        return {
            'performance_chart': fig_performance,
            'risk_return_chart': fig_risk_return,
            'metrics_table': metrics_df
        }
    
    def create_portfolio_analysis_section(self, returns_data):
        """Create portfolio optimization analysis."""
        optimizer = PortfolioOptimizer(returns_data)
        
        # Multiple optimization strategies
        strategies = {
            'Maximum Sharpe': optimizer.maximum_sharpe_optimization(),
            'Minimum Variance': optimizer.minimum_variance_optimization(),
            'Risk Parity': optimizer.risk_parity_optimization()
        }
        
        # Efficient frontier
        frontier = optimizer.efficient_frontier(n_points=100)
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Efficient Frontier with Optimal Portfolios',
                'Portfolio Weights Comparison',
                'Risk Decomposition',
                'Strategy Performance Metrics'
            ]
        )
        
        # Efficient frontier
        fig.add_trace(
            go.Scatter(
                x=frontier['volatility'],
                y=frontier['return'],
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='blue', width=3)
            ),
            row=1, col=1
        )
        
        # Optimal portfolios on frontier
        colors = ['red', 'green', 'orange']
        for i, (name, result) in enumerate(strategies.items()):
            if result['status'] == 'optimal':
                fig.add_trace(
                    go.Scatter(
                        x=[result['volatility']],
                        y=[result['expected_return']],
                        mode='markers',
                        name=name,
                        marker=dict(size=15, color=colors[i])
                    ),
                    row=1, col=1
                )
        
        # Portfolio weights
        weights_data = []
        for name, result in strategies.items():
            if result['status'] == 'optimal':
                for asset, weight in result['weights'].items():
                    weights_data.append({
                        'Strategy': name,
                        'Asset': asset,
                        'Weight': weight
                    })
        
        if weights_data:
            weights_df = pd.DataFrame(weights_data)
            for strategy in weights_df['Strategy'].unique():
                strategy_data = weights_df[weights_df['Strategy'] == strategy]
                fig.add_trace(
                    go.Bar(
                        x=strategy_data['Asset'],
                        y=strategy_data['Weight'],
                        name=f'{strategy} Weights',
                        opacity=0.7
                    ),
                    row=1, col=2
                )
        
        # Risk decomposition for one portfolio
        if strategies['Risk Parity']['status'] == 'optimal':
            rp_summary = optimizer.get_portfolio_summary(strategies['Risk Parity']['weights'])
            fig.add_trace(
                go.Bar(
                    x=rp_summary['risk_contributions'].index,
                    y=rp_summary['risk_contributions'].values,
                    name='Risk Contributions',
                    marker_color='purple'
                ),
                row=2, col=1
            )
        
        # Performance metrics
        perf_data = []
        for name, result in strategies.items():
            if result['status'] == 'optimal':
                perf_data.append({
                    'Strategy': name,
                    'Return': result['expected_return'],
                    'Volatility': result['volatility'],
                    'Sharpe': result['sharpe_ratio']
                })
        
        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            fig.add_trace(
                go.Bar(
                    x=perf_df['Strategy'],
                    y=perf_df['Sharpe'],
                    name='Sharpe Ratios',
                    marker_color='darkblue'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Portfolio Optimization Analysis',
            height=800,
            template='plotly_white'
        )
        
        return {
            'optimization_chart': fig,
            'strategies': strategies,
            'efficient_frontier': frontier
        }
    
    def create_risk_analysis_section(self, returns_data):
        """Create comprehensive risk analysis."""
        risk_metrics = RiskMetrics(returns_data)
        risk_report = risk_metrics.generate_risk_report()
        
        # Create risk dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Value at Risk (VaR) 95%',
                'Conditional VaR (CVaR) 95%',
                'Maximum Drawdown',
                'Sharpe Ratios',
                'Volatility Analysis',
                'Tail Risk Metrics'
            ]
        )
        
        assets = list(risk_report['var_estimates']['historical_95'].keys())
        
        # VaR 95%
        var_values = [risk_report['var_estimates']['historical_95'][asset] for asset in assets]
        fig.add_trace(
            go.Bar(x=assets, y=var_values, name='VaR 95%', marker_color='red'),
            row=1, col=1
        )
        
        # CVaR 95%
        cvar_values = [risk_report['cvar_estimates']['historical_95'][asset] for asset in assets]
        fig.add_trace(
            go.Bar(x=assets, y=cvar_values, name='CVaR 95%', marker_color='darkred'),
            row=1, col=2
        )
        
        # Maximum Drawdown
        dd_values = [risk_report['drawdown_metrics'][asset]['max_drawdown'] for asset in assets]
        fig.add_trace(
            go.Bar(x=assets, y=dd_values, name='Max DD', marker_color='orange'),
            row=1, col=3
        )
        
        # Sharpe Ratios
        sharpe_values = [risk_report['performance_metrics'][asset]['sharpe_ratio'] for asset in assets]
        fig.add_trace(
            go.Bar(x=assets, y=sharpe_values, name='Sharpe', marker_color='green'),
            row=2, col=1
        )
        
        # Volatility Analysis
        vol_values = [risk_report['volatility_metrics'][asset]['annual_volatility'] for asset in assets]
        fig.add_trace(
            go.Bar(x=assets, y=vol_values, name='Volatility', marker_color='blue'),
            row=2, col=2
        )
        
        # Skewness (Tail Risk)
        skew_values = [risk_report['tail_risk_metrics'][asset]['skewness'] for asset in assets]
        fig.add_trace(
            go.Bar(x=assets, y=skew_values, name='Skewness', marker_color='purple'),
            row=2, col=3
        )
        
        fig.update_layout(
            title='Comprehensive Risk Analysis',
            height=600,
            template='plotly_white',
            showlegend=False
        )
        
        return {
            'risk_dashboard': fig,
            'risk_report': risk_report
        }
    
    def create_regime_analysis_section(self, returns_data):
        """Create market regime analysis."""
        # Use the first asset for regime detection
        asset_returns = returns_data.iloc[:, 0]
        regime_detector = RegimeDetector(asset_returns)
        
        regime_report = regime_detector.generate_regime_report()
        
        # Create regime visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Volatility Regimes Over Time',
                'Regime Transition Probabilities',
                'Regime Performance Statistics',
                'Market Stress Indicators'
            ]
        )
        
        # Volatility regimes
        if 'volatility_clustering' in regime_report and 'error' not in regime_report['volatility_clustering']:
            vol_regimes = regime_report['volatility_clustering']['regimes']
            regime_colors = {0: 'green', 1: 'yellow', 2: 'red'}
            
            cumulative_returns = asset_returns.cumsum()
            
            for regime in [0, 1, 2]:
                regime_mask = vol_regimes == regime
                regime_dates = vol_regimes[regime_mask].index
                regime_cumret = cumulative_returns[regime_mask]
                
                if len(regime_cumret) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=regime_dates,
                            y=regime_cumret,
                            mode='markers',
                            name=f'Regime {regime}',
                            marker=dict(color=regime_colors[regime], size=4)
                        ),
                        row=1, col=1
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
                    marker=dict(size=15, color=['green', 'yellow', 'red']),
                    name='Regime Stats'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Market Regime Analysis',
            height=600,
            template='plotly_white'
        )
        
        return {
            'regime_chart': fig,
            'regime_report': regime_report
        }
    
    def create_options_analysis_section(self):
        """Create options pricing analysis."""
        # Example options analysis for a theoretical stock
        bs_model = BlackScholesModel(
            spot_price=100,
            strike_price=100,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            volatility=0.2
        )
        
        # Generate option chain
        strikes = np.arange(80, 121, 1)
        option_chain = bs_model.option_chain(strikes)
        
        # Volatility surface
        strike_range = np.arange(80, 121, 5)
        time_range = np.arange(0.1, 1.1, 0.1)
        vol_surface = bs_model.volatility_surface(strike_range, time_range)
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Option Prices vs Strike',
                'Greeks Profile',
                'Implied Volatility Surface',
                'Option Sensitivities'
            ],
            specs=[[{}, {}], [{"type": "surface"}, {}]]
        )
        
        # Option prices
        fig.add_trace(
            go.Scatter(
                x=option_chain['strike'],
                y=option_chain['call_price'],
                mode='lines',
                name='Call Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=option_chain['strike'],
                y=option_chain['put_price'],
                mode='lines',
                name='Put Price',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Greeks
        fig.add_trace(
            go.Scatter(
                x=option_chain['strike'],
                y=option_chain['call_delta'],
                mode='lines',
                name='Delta',
                line=dict(color='green')
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=option_chain['strike'],
                y=option_chain['call_gamma'] * 100,
                mode='lines',
                name='Gamma (×100)',
                line=dict(color='orange')
            ),
            row=1, col=2
        )
        
        # Volatility surface
        vol_pivot = vol_surface.pivot(index='time_to_expiry', columns='strike', values='implied_volatility')
        
        fig.add_trace(
            go.Surface(
                x=vol_pivot.columns,
                y=vol_pivot.index,
                z=vol_pivot.values,
                colorscale='Viridis',
                name='Vol Surface'
            ),
            row=2, col=1
        )
        
        # Risk sensitivities comparison
        atm_greeks = bs_model.all_greeks('call')
        greek_names = list(atm_greeks.keys())
        greek_values = list(atm_greeks.values())
        
        fig.add_trace(
            go.Bar(
                x=greek_names,
                y=greek_values,
                name='ATM Greeks',
                marker_color='purple'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Options Analytics Dashboard',
            height=800,
            template='plotly_white'
        )
        
        return {
            'options_chart': fig,
            'option_chain': option_chain,
            'volatility_surface': vol_surface
        }
    
    def generate_html_report(self, save_path='comprehensive_financial_report.html'):
        """Generate a comprehensive HTML report."""
        print("Generating comprehensive financial analysis report...")
        print("=" * 60)
        
        # Fetch real market data
        print("📊 Fetching real market data...")
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'QQQ', 'GLD']
        market_report = self.market_data_manager.generate_market_report(symbols, period='1y')
        
        if 'error' in market_report:
            print("❌ Error fetching market data. Using synthetic data for demonstration.")
            # Generate synthetic data as fallback
            np.random.seed(42)
            dates = pd.date_range('2023-01-01', periods=252, freq='D')
            returns_data = pd.DataFrame(
                np.random.multivariate_normal([0]*len(symbols), np.eye(len(symbols))*0.0001, 252),
                index=dates,
                columns=symbols
            )
            market_report = {'data': {'returns': returns_data}, 'symbols': symbols, 'period': '1y'}
        
        returns_data = market_report['data']['returns']
        
        # Generate all analysis sections
        print("📈 Creating market overview...")
        market_overview = self.create_market_overview_section(market_report)
        
        print("🎯 Performing portfolio optimization...")
        portfolio_analysis = self.create_portfolio_analysis_section(returns_data)
        
        print("⚠️  Conducting risk analysis...")
        risk_analysis = self.create_risk_analysis_section(returns_data)
        
        print("🔄 Analyzing market regimes...")
        regime_analysis = self.create_regime_analysis_section(returns_data)
        
        print("📊 Creating options analysis...")
        options_analysis = self.create_options_analysis_section()
        
        print("📋 Generating executive summary...")
        executive_summary = self.generate_executive_summary(
            market_report, 
            portfolio_analysis.get('strategies'), 
            risk_analysis
        )
        
        # Create HTML report
        html_content = self._create_html_template(
            executive_summary,
            market_overview,
            portfolio_analysis,
            risk_analysis,
            regime_analysis,
            options_analysis
        )
        
        # Save report
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Comprehensive report saved to: {save_path}")
        print("\n" + "=" * 60)
        print("📊 FINANCIAL ANALYSIS REPORT SUMMARY")
        print("=" * 60)
        
        print(f"📅 Report Date: {executive_summary['report_date']}")
        print(f"📈 Analysis Period: {executive_summary['analysis_period']}")
        print(f"🎯 Assets Analyzed: {executive_summary['assets_analyzed']}")
        print("\n🔍 Key Findings:")
        for finding in executive_summary['key_findings']:
            print(f"  • {finding}")
        
        print("\n🚀 This report demonstrates advanced quantitative finance capabilities!")
        print("📁 Open the HTML file in your browser for interactive visualizations")
        
        return save_path
    
    def _create_html_template(self, summary, market, portfolio, risk, regime, options):
        """Create the HTML template for the report."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Financial Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .section {{
            margin: 40px 0;
        }}
        .section h2 {{
            color: #2c3e50;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .key-finding {{
            background-color: #ecf0f1;
            padding: 10px;
            border-left: 4px solid #e74c3c;
            margin: 10px 0;
        }}
        .chart-container {{
            margin: 20px 0;
            min-height: 400px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #dee2e6;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .disclaimer {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 5px;
            margin-top: 30px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Comprehensive Financial Analysis Report</h1>
            <p><strong>Generated:</strong> {summary['report_date']}</p>
            <p><strong>Analysis Period:</strong> {summary['analysis_period']} | <strong>Assets:</strong> {summary['assets_analyzed']}</p>
        </div>

        <div class="section">
            <h2>📋 Executive Summary</h2>
            <div class="key-findings">
                <h3>Key Findings:</h3>
                {''.join(f'<div class="key-finding">• {finding}</div>' for finding in summary['key_findings'])}
            </div>
        </div>

        <div class="section">
            <h2>📈 Market Overview</h2>
            <div class="chart-container" id="performance-chart"></div>
            <div class="chart-container" id="risk-return-chart"></div>
        </div>

        <div class="section">
            <h2>🎯 Portfolio Optimization</h2>
            <div class="chart-container" id="portfolio-chart"></div>
        </div>

        <div class="section">
            <h2>⚠️ Risk Analysis</h2>
            <div class="chart-container" id="risk-chart"></div>
        </div>

        <div class="section">
            <h2>🔄 Market Regime Analysis</h2>
            <div class="chart-container" id="regime-chart"></div>
        </div>

        <div class="section">
            <h2>📊 Options Analytics</h2>
            <div class="chart-container" id="options-chart"></div>
        </div>

        <div class="disclaimer">
            <strong>Disclaimer:</strong> This report is for educational and demonstration purposes only. 
            It showcases advanced quantitative finance techniques and should not be used for actual investment decisions. 
            Past performance does not guarantee future results. All investments carry risk.
        </div>
    </div>

    <script>
        // Render all Plotly charts
        {self._generate_chart_scripts(market, portfolio, risk, regime, options)}
    </script>
</body>
</html>
        """
    
    def _generate_chart_scripts(self, market, portfolio, risk, regime, options):
        """Generate JavaScript for rendering charts."""
        scripts = []
        
        # Market overview charts
        if 'performance_chart' in market:
            scripts.append(f"Plotly.newPlot('performance-chart', {market['performance_chart'].to_json()});")
        
        if 'risk_return_chart' in market:
            scripts.append(f"Plotly.newPlot('risk-return-chart', {market['risk_return_chart'].to_json()});")
        
        # Portfolio chart
        if 'optimization_chart' in portfolio:
            scripts.append(f"Plotly.newPlot('portfolio-chart', {portfolio['optimization_chart'].to_json()});")
        
        # Risk chart
        if 'risk_dashboard' in risk:
            scripts.append(f"Plotly.newPlot('risk-chart', {risk['risk_dashboard'].to_json()});")
        
        # Regime chart
        if 'regime_chart' in regime:
            scripts.append(f"Plotly.newPlot('regime-chart', {regime['regime_chart'].to_json()});")
        
        # Options chart
        if 'options_chart' in options:
            scripts.append(f"Plotly.newPlot('options-chart', {options['options_chart'].to_json()});")
        
        return '\\n'.join(scripts)


def main():
    """Generate comprehensive financial analysis report."""
    report_generator = ComprehensiveReportGenerator()
    report_path = report_generator.generate_html_report()
    
    print(f"\\n🎉 Report generation complete!")
    print(f"📄 Report saved to: {report_path}")
    print("🌐 Open the HTML file in your web browser to view the interactive report")


if __name__ == "__main__":
    main()
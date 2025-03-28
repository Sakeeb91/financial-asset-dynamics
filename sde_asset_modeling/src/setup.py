from setuptools import setup, find_packages

setup(
    name="sde_asset_modeling",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "yfinance>=0.1.70",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="SDE models for financial asset dynamics",
    keywords="finance, stochastic, differential equations, simulation",
    python_requires=">=3.8",
) 
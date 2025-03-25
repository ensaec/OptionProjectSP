# Option Pricing Project

This project implements various methods for pricing options and managing financial data using different algorithms and models. All files are independent and can be executed individually. Below is an overview of each file and its functionality.

## Main Interface

### OptionPricer.py
This is the main menu interface for running the different pricing models. It allows you to access all other scripts and choose the method you want to use. You can start by executing this script to explore the pricing models.

## Files Overview

### MonteCarloPricer.py → Optimized Monte Carlo Simulation
This file uses the Monte Carlo method to estimate the price of options. The Monte Carlo simulation is optimized for performance and can be used for complex derivatives pricing. It uses market parameters as input to compute the option price and Greeks.

### BlackScholesPricer.py → Black-Scholes Implementation
This file implements the Black-Scholes pricing model. It requires market parameters (e.g., spot price, strike price, volatility, risk-free rate, and time to maturity) to compute the option price and the Greeks (Delta, Gamma, Vega, Theta, Rho).

### BinomialTreePricer.py → Binomial Tree Pricing
This file implements the binomial tree method for option pricing. Like the Black-Scholes model, it calculates the option price based on the provided market parameters and can be used to price both American and European options.

### RiskFreeCurve.py → Yield Curve Management
This script handles the management of the risk-free rate curve. It allows users to input a date and a desired maturity to interpolate the interest rate. The output is a risk-free rate curve and the interpolated rates.

### VolatilitySurface.py → Volatility Interpolation
This script loads market data (historical volatility and options data) and generates a volatility surface. It performs interpolation to estimate the volatility for different strikes and maturities, providing a surface of volatility for pricing options.

### NeuralNetworkPricer.py → ML Approximation
This file uses machine learning techniques to approximate option prices based on input market parameters. By training a neural network on historical data, it can predict option prices and evaluate their performance.

### Benchmarking.py → Method Comparison
This script creates functions to compare the performance of different pricing methods (e.g., Monte Carlo, Black-Scholes, Binomial Tree). It allows you to evaluate the accuracy and speed of each method by benchmarking them on the same set of input data.

### Other files

These files are mostly illustrations or commits that don't take part in the project. It can contain initial commits or steps undertaken that were abandoned later on. Some programs were executed only once to create a clean dataframe like DataProcessingBeforeML which enabled to create a clean file for the project.

## How to Execute

Each file in this project can be executed independently. To run any of the scripts:

1. Clone the repository:
   ```bash
   git clone https://github.com/ensaec/OptionPricingProject.git

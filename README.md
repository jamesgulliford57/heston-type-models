# Heston-Type Models

## Overview
This Python package simulates numerical solutions to financial asset stochastic differential equations and prices European options. This work is WIP.

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/jamesgulliford57/heston-type-models.git
cd heston-type-models
pip install -r requirements.txt
```

## Configuration
Configuration file can be found in the config_files/ directory. Model and
simulation parameters can be varied within the files.
### Example Configuration
```bash
[run]
model_name = Heston
simulator_name = MilsteinSimulator

[model_params]
lmbda = 1.0
sigma = 0.5
xi = 1.0
rho = -0.5
risk_free_rate = 0.05

[simulation]
initial_value = [1.0, 0.16]
final_time = 10.0
discretisation_parameter = 10000
number_of_paths = 100

[output]
output_directory = output/heston/milstein_simulator/test
```
## Run
```bash
python run.py <config_file_path>
```
Sample analysis files take as input the directory where simulation samples and output parameter jsons are stored. Example usage:
```bash
python sample_analysis/plot_trajectory <output_directory_path>
```

## To-do
- Volatility surface
   - Save surface data to avoid having to run calculations each time. Load from file if it exists.

### New Features
- Strong error on black_scholes model
   - Average value of separation of generated samples from sim and geometric brownian motion
- Plot volatility smiles and reproduce expected shapes in different regimes
- Performance metrics
   - Strong and weak error
   - Convergence
- Convergence of pricing variance with number of paths and discretisation
- Add more models
- Parallelise
   - mpi4py slow. Consider running expensive parts in Fortranx
- First hitting time
- Variance reduction
   - IS
   - Control variates
- Returns and stationarity
   - Plot histogram of returns
   - Test for order of integration
   - JB test
- Compare prices and implied volatilities of different simulation methods on single plot
- Exactly solvable models for analytical results - geometric brownian motion (Black Scholes)
- How can we be sure this is all doing the right thing
   - Test time t distribution of SDEs with nicely behaved dynamics
   - Tests of anything except models can be performed on Black Scholes geo BM
      - Log-normally distributed time-t prices
   - Geo BM has exact solution perform KS test on distributions of simulated and analytic BS samples and use as convergence metric
   - To verify simulators test simulated vs analytical black scholes option prices and implied volatilities run multiple and show convergence
- Seem to get very different paths for different schemes is this expected?
- Distribution transport properties - source of extra validation?
- Jump-diffusion model
- Hull-White model (mean reverting rates)
- Add American options - expectation of discounted payoff for optimal exercise strategy

### New Directions
- Exact simulation of diffusions
   - Simulate BM skeleton and accept/reject based on ratio of measures
   - Almost certainly doable but need to ascertain the benefits of this approach
- Stochastic rate models
   - Pricing and discounting with variable rates
- Rough volatility models
   - Recent paper suggest fractional Brownian motion better models stock prices
   - Implement and compare with vanilla BM
- Neural SDEs
   - Learn drift and diffusion from real data
   - Use torchsde
- Greeks with automatic differentiation
   - Use Jax to calculate Greeks on the fly
- Multiple asset portfolios

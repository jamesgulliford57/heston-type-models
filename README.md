# Heston-Type Models

## Overview
This Python package simulates numerical solutions to stochastic differential equations describing financial asset prices 
and values European options.

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/jamesgulliford57/heston-type-models.git
cd heston-type-models
pip install -r requirements.txt
```

## Configuration
Configuration files can be found in the config_files/ directory. Model and simulator names must be provided in camel case.
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
python run.py <config_path>
```
Two files are saved to the output directory post-simulation: samples.npy containing simulated samples and params.json containing
model and simulation parameters. samples.npy is a dictionary {'time' : time_values, <state1> : state1_values, <state2> : state2_values, ...} 
for each component of the model state vector e.g. ['price'] for Black-Scholes, ['price', 'volatility'] for Heston. state1_values, state2_values... are
arrays with size (number_of_paths, discretisation_parameter).

Sample analysis files take as input the directory where simulation samples and output parameter jsons are stored.
```bash
python sample_analysis/plot_trajectory <output_directory_path>
```

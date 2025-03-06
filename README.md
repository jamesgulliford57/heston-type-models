# Heston-Type Models

## Overview
This package simulates stochastic volatility models and prices European options 

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/heston-type-models.git
cd heston-type-models
pip install -r requirements.txt (requirements tbc)
```

## Configuration 
Configuration file can be found in the config_files/ directory. Model and 
simulation parameters can be varied within the files.
### Example Configuration
```bash
[model]
model_name = black_scholes 

[model_params]
r = 0.2
q = 0.0
sigma = 0.2

[simulation]
init_value = 1.0
final_time = 10.0 
n = 10000
num_paths = 100
scheme = milstein

[output]
output_dir = data
do_timestamp = false

[workflow]
run_simulation = true
run_analysis = true

[option]
run_option_pricing = true
option_type = call
strike_price = 4
maturity = 5.0
```
## Run
```bash
python build.py --config config.json
```

## To-do
### Fixes
- Fix negative volatilities
- Add other models and ability to select chosen model in config file
- Stochastic Model should not be specific to dimensions and variables of Heston model. also atm only work in 2D. At the moment Milstein scheme is specific to Heston.
- Protect against nonsensical inputs
- Replace generic terms like S and V with expalantory terms like price and volatility in state vectors
- Add model parameters to class docstrings

### New Features
- Performance metrics
   - Strong and weak error
   - Convergence
- Volatility surface
- Parallelise
- Add time decorator
- First hitting time
- More choice of plots
- Plot line on stock price graph to show option being priced by price_option
- Use call-put parity to price put options
- Change config file to ini or add file explaining config json
- Exactly solvable models for analytical results - geometric brownian motion (Black Scholes)

### New Directions
- Exact simulation of diffusions
- Neural ODEs

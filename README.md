# Heston-Type Models

## Overview
This package simulates stochastic financial models and prices European options. 

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
model_name = heston 

[model_params]
lmbda = 1.0
sigma = 0.5
xi = 1.0
rho = -0.5
risk_free_rate = 0.05

[simulation]
init_value = [1.0, 0.16] 
final_time = 10.0 
n = 10000
num_paths = 100
scheme = euler

[output]
output_dir = data
do_timestamp = false

[workflow]
run_simulation = false
run_analysis = true

[option]
run_option_pricing = true
option_type = call
strike_price = 2.5
maturity = 5.0
```
## Run
```bash
python build.py --config config_file/<model_name>.ini
```

## To-do
### Fixes
- Fix negative volatilities
- Protect against nonsensical inputs

### New Features
- Performance metrics
   - Strong and weak error
   - Convergence
- Volatility surface
- Add more models
- Parallelise
- Add time decorator
- First hitting time
- Plotting
- Use call-put parity to price put options
- Exactly solvable models for analytical results - geometric brownian motion (Black Scholes)
- How can we be sure this is all doing the right thing
- Seem to get very different paths for different schemes is this expected?

### New Directions
- Exact simulation of diffusions
- Neural ODEs

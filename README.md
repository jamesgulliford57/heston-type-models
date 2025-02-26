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
Configuration can be set within the config.json file. 
### Example Configuration
```bash
{
    "model_params": {
        "r": 0.2,
        "lmbda": 1.0,
        "sigma": 0.5,
        "xi": 1.0,
        "rho": -0.5
    },

    "init_value": [1, 0.16],
    "T": 10.0,
    "n": 10000,
    "N": 100,
    "scheme": "Euler",

    "output_dir" : "data",
    "do_timestamp" : false,

    "run_simulation" : true,
    "run_analysis" : true, 
    "run_option_pricing" : true, 
    "strike_price": 4,
    "maturity": 5.0
    
}
```
## Run
```bash
python main.py --config config.json
```

## To-do
### Fixes
- Fix negative volatilities
- Add other models and ability to select chosen model in config file
- Stochastic Model should not be specific to dimensions and variables of Heston model. also atm only work in 2D. At the moment Milstein scheme is specific to Heston.
- Protect against nonsensical inputs

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

### New Directions
- Exact simulation of diffusions

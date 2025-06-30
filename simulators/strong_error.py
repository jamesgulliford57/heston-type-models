import numpy as np

def exact_solution_gbm(X0, mu, sigma, bm_paths, T):
    """
    Exact solution for geometric Brownian motion (GBM).
    
    Parameters:
    -----------
    X0 : float
        Initial value.
    mu : float
        Drift coefficient.
    sigma : float
        Diffusion coefficient.
    bm_paths : np.ndarray
        Array of Brownian motion paths, shape (num_paths, num_steps + 1).
    T : float
        Total time.
        
    Returns:
    --------
    X_exact : np.ndarray
        Exact solution at time T for each path.
    """
    # Final BM values at time T
    B_T = bm_paths[:, -1]
    return X0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * B_T)

def estimate_strong_error(milstein_simulator, model_params, X0, mu, sigma, T, n_steps, n_paths):
    """
    Estimate strong error of Milstein scheme via Monte Carlo.
    
    Parameters:
    -----------
    milstein_simulator : MilsteinSimulator instance
        Your Milstein simulator object.
    model_params : dict
        Parameters for the model (e.g. drift, diffusion).
    X0 : float
        Initial condition.
    mu : float
        Drift coefficient.
    sigma : float
        Diffusion coefficient.
    T : float
        Total time.
    n_steps : int
        Number of discretisation steps.
    n_paths : int
        Number of Monte Carlo paths.
    
    Returns:
    --------
    strong_error : float
        Estimated strong error at time T.
    """
    dt = T / n_steps
    strong_errors = []

    for _ in range(n_paths):
        # Generate Brownian increments
        dW = np.random.normal(0, np.sqrt(dt), n_steps)
        bm_path = np.insert(np.cumsum(dW), 0, 0)  # BM path with initial 0

        # Initialize path_samples array for simulator, shape (dim, n_steps+1)
        path_samples = np.zeros((1, n_steps + 1))
        path_samples[0, 0] = X0

        # Run Milstein sim_path method (you may need to adapt this call to your code)
        milstein_simulator.discretisation_parameter = n_steps + 1
        simulated_path = milstein_simulator.sim_path(path_samples, dt)

        X_milstein_T = simulated_path[0, -1]
        X_exact_T = X0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * bm_path[-1])

        strong_errors.append(abs(X_milstein_T - X_exact_T))

    return np.mean(strong_errors)

if __name__ == "__main__":
    from milstein_simulator import MilsteinSimulator
    from models.dupire_local_volatility import DupireLocalVolailtity

    # Example parameters
    X0 = 1.0
    mu = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 1000
    n_paths = 1000

    # Instantiate model and simulator
    model_params = {"risk_free_rate": mu, "sigma": sigma}  # adapt as needed
    model = YourModelClass(model_params)
    simulator_params = {"discretisation_parameter": n_steps + 1}
    simulator = MilsteinSimulator(model, simulator_params)

    # Estimate strong error
    strong_err = estimate_strong_error(simulator, model_params, X0, mu, sigma, T, n_steps, n_paths)
    print(f"Estimated strong error at T={T}: {strong_err}")

import numpy as np
import matplotlib.pyplot as plt

class HestonModel:
    def __init__(self, r, lmbda, sigma, ksi, rho, n, T, initial_value):
        self.lmbda = lmbda
        self.sigma = sigma
        self.ksi = ksi 
        self.rho = rho 
        self.r = r  
        self.T = T 
        self.initial_value = initial_value
        if not isinstance(n, int) or n < 1:
            raise ValueError(f'n must be an positive integer. Provided: {n}')
        else:
            self.n = n
        self.h = self.T / self.n
        self.hvalues = np.linspace(0, self.T, self.n)
        self.samples = np.zeros((2, self.n))
        self.samples[:, 0] = self.initial_value.flatten()

    def _mu_SDE(self, S, V):
        """
        Compute Heston model drift
        """
        return np.array([[self.r * S], [self.lmbda * (self.sigma ** 2 - V)]])

    def _sigma_SDE(self, S, V):
        """ 
        Compute Heston model volatility
        """
        return np.array([[S * np.sqrt(np.abs(V)), 0], [self.rho * self.ksi * np.sqrt(np.abs(V)), np.sqrt(1 - self.rho ** 2) * self.ksi * np.sqrt(np.abs(V))]])

    def _euler_maryuama_path(self):
        """
        Simulates one path of the Heston model using the Euler-Maruyama scheme to solve the 
        SDE defined by mu_SDE and sigma_SDE
        """
        self.samples = np.zeros((2, self.n)) # Reset samples arrays before each simulation
        self.samples[:, 0] = self.initial_value.flatten()
        # Simulate Heston model with Euler-Maruyama scheme
        for i in range(1, self.n):

            bm_1 = np.random.normal(0, np.sqrt(self.h))
            bm_2 = np.random.normal(0, np.sqrt(self.h))
            bm_step = np.array([[bm_1], [self.rho * bm_1 + np.sqrt(1 - self.rho ** 2) * bm_2]])
            self.samples[:, i] = self.samples[:, i - 1] + self._mu_SDE(self.samples[0, i - 1], self.samples[1, i - 1]).flatten() * self.h + np.dot(self._sigma_SDE(self.samples[0, i - 1], self.samples[1, i - 1]), bm_step).flatten()

    def price_option(self, K, N):
        """
        Simulates N Euler-Maruyama paths and compute average price for an option with strike price K
        and maturity T
        """
        C_values = np.zeros(N)

        for i in range(N):
            self._euler_maryuama_path()
            stock_prices = self.samples[0,:]
            C_values[i] = np.exp(-self.r * self.T) * max(stock_prices[-1] - K, 0)

        print(f'Option price: {np.mean(C_values):.2f}')
    
    def plot_trajectory(self, figname):
        """
        Plot single stock price and volatility path
        """
        self._euler_maryuama_path()
        plt.figure(figsize=(10, 10))
        plt.subplot(211)
        plt.plot(self.hvalues, self.samples[0])
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title("Stock price")
        plt.subplot(212)
        plt.plot(self.hvalues, self.samples[1])
        plt.xlabel('Time')
        plt.ylabel('Volatility')
        plt.title("Implied volatility")
        plt.tight_layout()
        plt.savefig(figname, dpi=300)

    def plot_multiple_trajectories(self, figname, N):
        """
        Plot N stock price and volatility paths on same figure
        """
        fig, (ax_price, ax_vol) = plt.subplots(2, 1, figsize=(10, 8))
        for i in range(N):
            self._euler_maryuama_path()
            ax_price.plot(self.hvalues, self.samples[0], alpha = 0.7)
            ax_vol.plot(self.hvalues, self.samples[1], alpha = 0.7)
            print(f'Simulation {i + 1} complete')
        print(f'All simulations complete. Trajectories plotted in {figname}')
        ax_price.set_xlabel("Time")
        ax_price.set_ylabel("Price")
        ax_price.set_title("Stock Price")

        ax_vol.set_xlabel("Time")
        ax_vol.set_ylabel("Volatility")
        ax_vol.set_title("Implied Volatility")

        fig.tight_layout()
        plt.savefig(figname, dpi=300)


heston1 = HestonModel(r=0.2, lmbda=1, sigma=0.5, ksi=1, 
                      rho=-0.5, n=10000, T=5, initial_value=np.array([[1], [0.16]]))
#heston1.price_option(K=2, N=200)
heston1.plot_multiple_trajectories('heston.png', N=5)

'''
To add:
i) Find option prices across strike prices to plot implied volatility smile
ii) Ask chatGPT for project ideas

'''
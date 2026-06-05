import numpy as np

class CIRModel:
    def __init__(self, kappa, theta, sigma):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        
    def get_A_B(self, tau):
        h = np.sqrt(self.kappa**2 + 2 * self.sigma**2)
        exp_h = np.exp(h * tau)
        num_A = 2 * h * np.exp((self.kappa + h) * tau / 2)
        den_A = 2 * h + (self.kappa + h) * (exp_h - 1)
        A = (num_A / den_A)**(2 * self.kappa * self.theta / self.sigma**2)
        B = 2 * (exp_h - 1) / den_A
        return A, B
        
    def compute_yields(self, r_t, tau):
        A, B = self.get_A_B(tau)
        if isinstance(r_t, np.ndarray) and isinstance(tau, np.ndarray):
            r_col = r_t[:, np.newaxis]
            tau_row = tau[np.newaxis, :]
            A_row = A[np.newaxis, :]
            B_row = B[np.newaxis, :]
            return (B_row * r_col - np.log(A_row)) / tau_row
        else:
            return (B * r_t - np.log(A)) / tau

    def simulate_exact(self, r0, T, n_steps, n_paths=1):
        dt = T / n_steps
        r = np.zeros((n_steps + 1, n_paths))
        r[0] = r0
        exp_k = np.exp(-self.kappa * dt)
        c = 2 * self.kappa / (self.sigma**2 * (1 - exp_k))
        df = 4 * self.kappa * self.theta / self.sigma**2
        for t in range(1, n_steps + 1):
            nc = 2 * c * r[t-1] * exp_k
            chi_val = np.random.noncentral_chisquare(df, nc, size=(n_paths,))
            r[t] = chi_val / (2 * c)
        return r

import numpy as np

class CIRTwoFactorModel:
    def __init__(self, kx, tx, sx, ky, ty, sy, lam):
        self.kx = kx
        self.tx = tx
        self.sx = sx
        self.ky = ky
        self.ty = ty
        self.sy = sy
        self.lam = lam
        
    def get_A_B_factor(self, kappa, theta, sigma, tau):
        h = np.sqrt(kappa**2 + 2 * sigma**2)
        exp_h = np.exp(h * tau)
        num_A = 2 * h * np.exp((kappa + h) * tau / 2)
        den_A = 2 * h + (kappa + h) * (exp_h - 1)
        A = (num_A / den_A)**(2 * kappa * theta / sigma**2)
        B = 2 * (exp_h - 1) / den_A
        return A, B
        
    def run_state_filter(self, r_series):
        N = len(r_series)
        x = np.zeros(N)
        x[0] = r_series[0]
        for t in range(1, N):
            x[t] = (1.0 - self.lam) * x[t-1] + self.lam * r_series[t]
        y = r_series - x
        return x, y
        
    def compute_yields(self, x_t, y_t, taus):
        Ax, Bx = self.get_A_B_factor(self.kx, self.tx, self.sx, taus)
        Ay, By = self.get_A_B_factor(self.ky, self.ty, self.sy, taus)
        if isinstance(x_t, np.ndarray):
            x_col = x_t[:, np.newaxis]
            y_col = y_t[:, np.newaxis]
            tau_row = taus[np.newaxis, :]
            term_r = Bx[np.newaxis, :] * x_col + By[np.newaxis, :] * y_col
            term_const = -np.log(Ax * Ay)[np.newaxis, :]
            return (term_r + term_const) / tau_row
        else:
            return (Bx * x_t + By * y_t - np.log(Ax * Ay)) / taus

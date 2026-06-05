import numpy as np
from scipy.optimize import minimize
from scipy.stats import ncx2
from src.cir_model import CIRModel
from src.two_factor_cir import CIRTwoFactorModel

class CIRCalibration:
    def __init__(self, r_all, Y_all, train_idx, taus, dt):
        self.r_all = r_all
        self.Y_all = Y_all
        self.train_idx = train_idx
        self.taus = taus
        self.dt = dt
        
    def calibrate_ols(self):
        dr = np.diff(self.r_all)
        r_t = self.r_all[:-1]
        Y_w = dr / np.sqrt(r_t)
        X1_w = 1.0 / np.sqrt(r_t)
        X2_w = np.sqrt(r_t)
        X_w = np.column_stack((X1_w, X2_w))
        beta_w, _, _, _ = np.linalg.lstsq(X_w, Y_w, rcond=None)
        beta1, beta2 = beta_w
        kappa = -beta2 / self.dt
        theta = beta1 / (kappa * self.dt)
        residuals = Y_w - X_w.dot(beta_w)
        sigma = np.sqrt(np.var(residuals) / self.dt)
        return kappa, theta, sigma
        
    def calibrate_mle(self):
        def cir_neg_log_lik(params):
            k, th, sig = params
            if k <= 0 or th <= 0 or sig <= 0:
                return 1e10
            exp_k = np.exp(-k * self.dt)
            c = 2 * k / (sig**2 * (1 - exp_k))
            df = 4 * k * th / sig**2
            nc = 2 * c * self.r_all[:-1] * exp_k
            x = 2 * c * self.r_all[1:]
            logpdf = ncx2.logpdf(x, df, nc)
            log_lik = logpdf + np.log(2 * c)
            if np.any(np.isnan(log_lik)) or np.any(np.isinf(log_lik)):
                return 1e10
            return -np.sum(log_lik)
        res = minimize(cir_neg_log_lik, x0=[0.1, 0.02, 0.02], bounds=((1e-4, 50.0), (1e-4, 2.0), (1e-4, 1.0)), method='L-BFGS-B', options={'maxiter': 500})
        return res.x[0], res.x[1], res.x[2]
        
    def calibrate_cross_sectional(self):
        def objective(params):
            k, th, sig = params
            if k <= 0 or th <= 0 or sig <= 0:
                return 1e10
            model = CIRModel(k, th, sig)
            preds = model.compute_yields(self.r_all[self.train_idx], self.taus)
            return np.sum((self.Y_all[self.train_idx] - preds)**2)
        res = minimize(objective, x0=[0.1, 0.02, 0.02], bounds=((1e-4, 5.0), (1e-4, 0.2), (1e-4, 0.2)), method='L-BFGS-B', options={'maxiter': 500})
        return res.x[0], res.x[1], res.x[2]
        
    def calibrate_two_factor(self):
        def objective(params):
            kx, tx, sx, ky, ty, sy, lam = params
            if kx <= 0 or tx <= 0 or sx <= 0 or ky <= 0 or ty <= 0 or sy <= 0 or lam < 0.001 or lam > 0.99:
                return 1e12
            if 2*kx*tx < sx**2 or 2*ky*ty < sy**2:
                return 1e12
            model = CIRTwoFactorModel(kx, tx, sx, ky, ty, sy, lam)
            x_est, y_est = model.run_state_filter(self.r_all)
            x_col = x_est[self.train_idx, np.newaxis]
            y_col = y_est[self.train_idx, np.newaxis]
            tau_row = self.taus[np.newaxis, :]
            Ax, Bx = model.get_A_B_factor(kx, tx, sx, self.taus)
            Ay, By = model.get_A_B_factor(ky, ty, sy, self.taus)
            preds = (Bx[np.newaxis, :] * x_col + By[np.newaxis, :] * y_col - np.log(Ax * Ay)[np.newaxis, :]) / tau_row
            mse = np.mean((self.Y_all[self.train_idx] - preds)**2)
            penalty = 1e-3 * (sx**2 + sy**2) + 1e-4 * (kx - 0.16)**2
            return mse + penalty
        x0 = [0.16, 0.02, 0.001, 0.16, 0.02, 0.001, 0.05]
        bounds = ((0.01, 1.0), (0.001, 0.1), (1e-6, 0.005),
                  (0.01, 1.0), (0.001, 0.1), (1e-6, 0.005),
                  (0.001, 0.5))
        res = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 500})
        return res.x

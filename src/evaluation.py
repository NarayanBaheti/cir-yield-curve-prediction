import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class CIREvaluator:
    @staticmethod
    def compute_metrics(actual, predicted, maturities):
        metrics_list = []
        for idx, col in enumerate(maturities):
            act = actual[:, idx]
            pr = predicted[:, idx]
            r2 = r2_score(act, pr)
            rmse = np.sqrt(mean_squared_error(act, pr))
            mae = mean_absolute_error(act, pr)
            mape = np.mean(np.abs((act - pr) / act)) * 100
            mbe = np.mean(pr - act)
            metrics_list.append([col, r2, rmse, mae, mape, mbe])
        df = pd.DataFrame(metrics_list, columns=['Maturity', 'R2', 'RMSE', 'MAE', 'MAPE (%)', 'MBE'])
        
        overall_r2 = r2_score(actual.flatten(), predicted.flatten())
        overall_rmse = np.sqrt(mean_squared_error(actual.flatten(), predicted.flatten()))
        overall_mae = mean_absolute_error(actual.flatten(), predicted.flatten())
        overall_mape = np.mean(np.abs((actual.flatten() - predicted.flatten()) / actual.flatten())) * 100
        overall_mbe = np.mean(predicted.flatten() - actual.flatten())
        overall_stats = {'R2': overall_r2, 'RMSE': overall_rmse, 'MAE': overall_mae, 'MAPE (%)': overall_mape, 'MBE': overall_mbe}
        return df, overall_stats

class CIRBacktester:
    def __init__(self, model_base, model_2f):
        self.model_base = model_base
        self.model_2f = model_2f
        
    def backtest_base(self, r_series, taus):
        return self.model_base.compute_yields(r_series, taus)
        
    def backtest_two_factor(self, r_full_series, target_idx, taus):
        x_full, y_full = self.model_2f.run_state_filter(r_full_series)
        x_target = x_full[target_idx]
        y_target = y_full[target_idx]
        return self.model_2f.compute_yields(x_target, y_target, taus), x_target, y_target

import numpy as np

def run_ewma_filter(r_series, lam):
    """
    Exponentially Weighted Moving Average Filter as a steady-state Kalman Filter representation
    to decompose the short rate into independent short-term (x_t) and long-term (y_t) latent factors.
    """
    N = len(r_series)
    x = np.zeros(N)
    x[0] = r_series[0]
    for t in range(1, N):
        x[t] = (1.0 - lam) * x[t-1] + lam * r_series[t]
    y = r_series - x
    return x, y

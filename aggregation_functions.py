import numpy as np

FUNCTIONS = {
    'mean': np.mean,
    'median': np.median,
    'std': np.std,
    'min': np.min,
    'max': np.max,
    'ptp': np.ptp,
    'energy': lambda x: np.sum(x ** 2),
    'rms': lambda x: np.sqrt(np.sum(x ** 2) / len(x)),
    'lineintegral': lambda x: np.abs(np.diff(x)).sum(),
    'n_sign_changes': lambda x: np.sum(np.diff(np.sign(x)) != 0),
    'iqr': lambda x: np.subtract(*np.nanpercentile(x, [75, 25])),
    'iqr_5_95': lambda x: np.subtract(*np.nanpercentile(x, [95, 5])),
    'pct_5': lambda x: np.percentile(x, 5),
    'pct_95': lambda x: np.percentile(x, 95)
}

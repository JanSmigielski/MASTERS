import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.kernel_approximation import RBFSampler
from scipy.stats import skew
import statsmodels.api as sm

def simple_model(predictors, target_series, target_name, window_size=12):
    records = []
    X = predictors.values
    y = target_series.values

    for t in tqdm(range(window_size + 2, len(target_series))):
        X_window = X[t - window_size - 1 : t - 1]
        y_window = y[t - window_size : t].reshape(-1, 1)

        Xt = X_window.T
        try:
            XtX_inv = np.linalg.inv(X_window @ Xt)
        except np.linalg.LinAlgError:
            continue  # Skip if matrix is not invertible

        beta = Xt @ XtX_inv @ y_window
        x_t_minus_1 = X[t - 1].reshape(1, -1)
        predicted = float((x_t_minus_1 @ beta).item())  # force scalar
        actual = float(y[t])
        strategy = predicted * actual

        records.append({
            'date': target_series.index[t],
            'predicted return': predicted,
            f'{target_name}': actual,
            'strategy return': strategy
        })

    return pd.DataFrame(records).set_index('date')


def complex_model(predictors, target_series, target_name, window_size=12, alpha=1e3, gamma=4, n_components=12000):
    records = []
    model = Ridge(alpha=alpha, solver='svd')
    rbf_feature = RBFSampler(gamma=gamma, n_components=n_components, random_state=42)
    transformed_X = rbf_feature.fit_transform(predictors)
    y = target_series.values

    for t in tqdm(range(window_size + 2, len(target_series))):
        X_window = transformed_X[t - window_size - 1 : t - 1]
        y_window = y[t - window_size : t]

        model.fit(X_window, y_window)
        x_t_minus_1 = transformed_X[t - 1:t]
        predicted = float(model.predict(x_t_minus_1)[0])  # force scalar
        actual = float(y[t])
        strategy = predicted * actual

        records.append({
            'date': target_series.index[t],
            'predicted return': predicted,
            f'{target_name}': actual,
            'strategy return': strategy
        })

    return pd.DataFrame(records).set_index('date')


def performance_metrics(records, target_name):
    records['strategy return'] = records['strategy return'].apply(lambda x: float(x[0]) if isinstance(x, (np.ndarray, list)) else float(x))
    records[target_name] = records[target_name].apply(lambda x: float(x[0]) if isinstance(x, (np.ndarray, list)) else float(x))

    excess_returns = records['strategy return']
    sharpe = (excess_returns.mean() * 12) / (excess_returns.std() * np.sqrt(12))

    X = sm.add_constant(records[target_name])
    y = excess_returns
    model = sm.OLS(y, X).fit()

    alpha = model.params[0]
    tracking_error = model.resid.std()
    appraisal_ratio = alpha / tracking_error
    alpha_t_stat = model.tvalues[0]
    skewness = skew(excess_returns)

    return {
        'Sharpe Ratio': sharpe,
        'Appraisal Ratio': appraisal_ratio,
        'Alpha T-stat': alpha_t_stat,
        'Skewness': skewness
    }
    
def factor_stats(target_series):
    sharpe_ratio = (target_series.mean() * 12) / (target_series.std() * np.sqrt(12))
    skewness = skew(target_series.dropna())
    return {
        'factor_sharpe': sharpe_ratio,
        'factor_skew': skewness
    }

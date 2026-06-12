import pandas as pd
import numpy as np

def compute_var(returns, confidence=0.95):
    """Value at Risk at given confidence level."""
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))

def compute_cvar(returns, confidence=0.95):
    """Conditional Value at Risk (Expected Shortfall)."""
    var = compute_var(returns, confidence)
    return float(returns[returns <= var].mean())

def compute_risk_metrics(df, ticker, risk_free=0.06):
    """Full risk report for a single stock."""
    stock = df[df['ticker'] == ticker].copy().sort_values('date')
    ret   = stock['daily_return'].dropna()

    annual_return = ret.mean() * 252
    annual_vol    = ret.std()  * np.sqrt(252)
    sharpe        = (annual_return - risk_free) / (annual_vol + 1e-10)
    downside      = ret[ret < 0].std() * np.sqrt(252)
    sortino       = (annual_return - risk_free) / (downside + 1e-10)

    cumulative    = (1 + ret).cumprod()
    rolling_max   = cumulative.cummax()
    drawdown      = (cumulative - rolling_max) / (rolling_max + 1e-10)
    max_drawdown  = drawdown.min()

    var_95  = compute_var(ret, 0.95)
    var_99  = compute_var(ret, 0.99)
    cvar_95 = compute_cvar(ret, 0.95)

    return {
        'ticker'        : ticker,
        'annual_return' : round(annual_return * 100, 2),
        'annual_vol'    : round(annual_vol    * 100, 2),
        'sharpe_ratio'  : round(sharpe,  2),
        'sortino_ratio' : round(sortino, 2),
        'max_drawdown'  : round(max_drawdown * 100, 2),
        'var_95'        : round(var_95  * 100, 2),
        'var_99'        : round(var_99  * 100, 2),
        'cvar_95'       : round(cvar_95 * 100, 2),
        'skewness'      : round(float(ret.skew()), 4),
        'kurtosis'      : round(float(ret.kurt()), 4),
    }

def compute_rolling_sharpe(returns, window=252, risk_free=0.06):
    """Rolling Sharpe Ratio series."""
    rolling_ret = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std()  * np.sqrt(252)
    return (rolling_ret - risk_free) / (rolling_vol + 1e-10)

def compute_drawdown_series(returns):
    """Full drawdown series as percentage."""
    cumulative  = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / (rolling_max + 1e-10)
    return drawdown * 100

def full_risk_report(df, risk_free=0.06):
    """Risk metrics for all stocks — returns a DataFrame."""
    results = []
    for ticker in df['ticker'].unique():
        try:
            metrics = compute_risk_metrics(df, ticker, risk_free)
            results.append(metrics)
        except Exception:
            continue
    return pd.DataFrame(results).sort_values('sharpe_ratio', ascending=False)
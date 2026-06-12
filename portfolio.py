import pandas as pd
import numpy as np

def compute_stock_metrics(df, risk_free=0.06):
    """Compute risk/return metrics for every stock."""
    results = []
    for ticker in df['ticker'].unique():
        stock = df[df['ticker'] == ticker].copy()
        ret   = stock['daily_return'].dropna()
        if len(ret) < 50:
            continue
        annual_return = ret.mean() * 252
        annual_vol    = ret.std()  * np.sqrt(252)
        sharpe        = (annual_return - risk_free) / (annual_vol + 1e-10)
        downside      = ret[ret < 0].std() * np.sqrt(252)
        sortino       = (annual_return - risk_free) / (downside + 1e-10)
        cumulative    = (1 + ret).cumprod()
        rolling_max   = cumulative.cummax()
        drawdown      = (cumulative - rolling_max) / (rolling_max + 1e-10)
        max_drawdown  = drawdown.min()
        results.append({
            'ticker'       : ticker,
            'annual_return': annual_return,
            'annual_vol'   : annual_vol,
            'sharpe_ratio' : sharpe,
            'sortino_ratio': sortino,
            'max_drawdown' : max_drawdown,
            'num_days'     : len(ret)
        })
    return pd.DataFrame(results)

def build_portfolio(metrics_df, profile='balanced'):
    """
    Build a portfolio for a given investor profile.
    Profiles: 'conservative', 'balanced', 'aggressive'
    Returns DataFrame with ticker weights and metrics.
    """
    df = metrics_df.copy()

    if profile == 'conservative':
        df['score'] = (
            -df['annual_vol']    * 0.5 +
             df['annual_return'] * 0.2 +
            -df['max_drawdown']  * 0.3
        )
        top_n = 8

    elif profile == 'balanced':
        df['score'] = (
            df['sharpe_ratio']  * 0.5 +
            df['annual_return'] * 0.3 +
           -df['annual_vol']    * 0.2
        )
        top_n = 10

    elif profile == 'aggressive':
        df['score'] = (
            df['annual_return'] * 0.6 +
            df['sharpe_ratio']  * 0.2 +
            df['sortino_ratio'] * 0.2
        )
        top_n = 12

    selected = df.nlargest(top_n, 'score').copy()
    selected['score_shifted'] = selected['score'] - selected['score'].min() + 0.01
    selected['weight']        = selected['score_shifted'] / selected['score_shifted'].sum()
    selected['profile']       = profile

    return selected[['ticker','weight','annual_return','annual_vol',
                      'sharpe_ratio','max_drawdown','profile']]

def simulate_portfolio(portfolio_df, returns_pivot):
    """
    Simulate portfolio performance over time.
    Returns cumulative return series and summary stats.
    """
    tickers = portfolio_df['ticker'].tolist()
    weights = portfolio_df['weight'].values
    valid   = [(t, w) for t, w in zip(tickers, weights)
               if t in returns_pivot.columns]
    tickers = [x[0] for x in valid]
    weights = np.array([x[1] for x in valid])
    weights = weights / weights.sum()

    port_returns = returns_pivot[tickers].dot(weights)
    cumulative   = (1 + port_returns).cumprod()
    annual_ret   = port_returns.mean() * 252
    annual_vol   = port_returns.std()  * np.sqrt(252)
    sharpe       = (annual_ret - 0.06) / (annual_vol + 1e-10)
    max_dd       = ((cumulative - cumulative.cummax()) / cumulative.cummax()).min()
    total_ret    = cumulative.iloc[-1] - 1

    stats = {
        'total_return'   : round(total_ret * 100, 2),
        'annual_return'  : round(annual_ret * 100, 2),
        'annual_vol'     : round(annual_vol * 100, 2),
        'sharpe_ratio'   : round(sharpe, 2),
        'max_drawdown'   : round(max_dd * 100, 2),
    }
    return cumulative, stats
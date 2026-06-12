import pandas as pd
import numpy as np
import joblib
import os

FEATURES = [
    'MA_7','MA_21','MA_50','EMA_12','EMA_26',
    'MACD','MACD_signal','MACD_hist',
    'RSI','BB_width','BB_pos',
    'momentum_5','momentum_10','momentum_21',
    'volatility_21','volume_ratio',
    'hl_spread','close_vs_open',
    'price_vs_MA50','price_vs_MA200'
]

def add_technical_indicators(stock_df):
    d = stock_df.copy().reset_index(drop=True)
    d['MA_7']        = d['close'].rolling(7).mean()
    d['MA_21']       = d['close'].rolling(21).mean()
    d['MA_50']       = d['close'].rolling(50).mean()
    d['MA_200']      = d['close'].rolling(200).mean()
    d['EMA_12']      = d['close'].ewm(span=12, adjust=False).mean()
    d['EMA_26']      = d['close'].ewm(span=26, adjust=False).mean()
    d['MACD']        = d['EMA_12'] - d['EMA_26']
    d['MACD_signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['MACD_hist']   = d['MACD'] - d['MACD_signal']
    delta            = d['close'].diff()
    gain             = delta.clip(lower=0)
    loss             = -delta.clip(upper=0)
    avg_gain         = gain.rolling(14).mean()
    avg_loss         = loss.rolling(14).mean()
    rs               = avg_gain / (avg_loss + 1e-10)
    d['RSI']         = 100 - (100 / (1 + rs))
    d['BB_mid']      = d['close'].rolling(20).mean()
    d['BB_std']      = d['close'].rolling(20).std()
    d['BB_upper']    = d['BB_mid'] + 2 * d['BB_std']
    d['BB_lower']    = d['BB_mid'] - 2 * d['BB_std']
    d['BB_width']    = (d['BB_upper'] - d['BB_lower']) / (d['BB_mid'] + 1e-10)
    d['BB_pos']      = (d['close'] - d['BB_lower']) / (d['BB_upper'] - d['BB_lower'] + 1e-10)
    d['momentum_5']  = d['close'].pct_change(5)
    d['momentum_10'] = d['close'].pct_change(10)
    d['momentum_21'] = d['close'].pct_change(21)
    d['volatility_21']  = d['close'].pct_change().rolling(21).std()
    d['volume_MA_10']   = d['volume'].rolling(10).mean()
    d['volume_ratio']   = d['volume'] / (d['volume_MA_10'] + 1e-10)
    d['daily_return']   = d['close'].pct_change()
    d['hl_spread']      = (d['high'] - d['low']) / (d['close'] + 1e-10)
    d['close_vs_open']  = (d['close'] - d['open']) / (d['open'] + 1e-10)
    d['price_vs_MA50']  = (d['close'] - d['MA_50'])  / (d['MA_50']  + 1e-10)
    d['price_vs_MA200'] = (d['close'] - d['MA_200']) / (d['MA_200'] + 1e-10)
    d['target']         = (d['close'].shift(-1) > d['close']).astype(int)
    return d

def predict_stock(ticker, df, model):
    """
    Given a ticker and full dataframe, returns:
    - prediction (0=DOWN, 1=UP)
    - probability of UP
    - latest technical signals
    """
    stock      = df[df['ticker'] == ticker].sort_values('date')
    stock      = add_technical_indicators(stock)
    latest_row = stock.dropna(subset=FEATURES).iloc[-1]

    X          = pd.DataFrame([latest_row[FEATURES].values], columns=FEATURES)
    pred       = model.predict(X)[0]
    proba      = model.predict_proba(X)[0]

    signals = {
        'ticker'        : ticker,
        'date'          : latest_row['date'],
        'close'         : latest_row['close'],
        'prediction'    : 'UP' if pred == 1 else 'DOWN',
        'confidence'    : round(proba[pred] * 100, 2),
        'prob_up'       : round(proba[1] * 100, 2),
        'prob_down'     : round(proba[0] * 100, 2),
        'RSI'           : round(latest_row['RSI'], 2),
        'MACD'          : round(latest_row['MACD'], 4),
        'BB_pos'        : round(latest_row['BB_pos'], 4),
        'momentum_5'    : round(latest_row['momentum_5'], 4),
        'price_vs_MA50' : round(latest_row['price_vs_MA50'], 4),
    }
    return signals

def predict_all_stocks(df, model):
    """Returns prediction for every stock as a DataFrame."""
    results = []
    for ticker in df['ticker'].unique():
        try:
            sig = predict_stock(ticker, df, model)
            results.append(sig)
        except Exception:
            continue
    return pd.DataFrame(results).sort_values('prob_up', ascending=False)

def load_model(model_path):
    return joblib.load(model_path)
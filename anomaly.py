import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ANOMALY_FEATURES = [
    'daily_return', 'volume_ratio', 'hl_spread',
    'volatility_21', 'momentum_5', 'close_vs_open'
]

def detect_anomalies(df, contamination=0.02):
    """
    Run Isolation Forest anomaly detection on full dataset.
    Returns df with anomaly labels and rule-based type flags.
    """
    df_out = df.copy()
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]

    available = [f for f in ANOMALY_FEATURES if f in df_out.columns]
    df_clean  = df_out.dropna(subset=available).copy()

    scaler     = StandardScaler()
    X          = scaler.fit_transform(df_clean[available])
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    df_clean['anomaly_score'] = iso_forest.fit_predict(X)
    df_clean['anomaly_raw']   = iso_forest.decision_function(X)
    df_clean['is_anomaly']    = (df_clean['anomaly_score'] == -1).astype(int)

    # Rule-based labels
    df_clean['volatility_spike'] = (
        df_clean['daily_return'].abs() >
        df_clean['daily_return'].abs().quantile(0.99)
    ).astype(int)

    df_clean['volume_spike'] = (
        df_clean['volume_ratio'] >
        df_clean['volume_ratio'].quantile(0.99)
    ).astype(int)

    df_clean['extreme_crash'] = (
        df_clean['daily_return'] ,
        df_clean['daily_return'].quantile(0.005)
    ).astype(int)

    df_clean['extreme_surge'] = (
        df_clean['daily_return'] >
        df_clean['daily_return'].quantile(0.995)
    ).astype(int)

    return df_clean

def get_top_anomalies(anomaly_df, n=20):
    """Return the n most extreme anomalies."""
    return (anomaly_df[anomaly_df['is_anomaly'] == 1]
            .sort_values('anomaly_raw')
            .head(n)[['date','ticker','close','daily_return',
                       'volume_ratio','volatility_spike',
                       'volume_spike','extreme_crash','extreme_surge']]
            .reset_index(drop=True))

def get_market_anomaly_timeline(anomaly_df):
    """Count anomalies per date across all stocks."""
    timeline = (anomaly_df[anomaly_df['is_anomaly'] == 1]
                .groupby('date').size()
                .reset_index())
    timeline.columns = ['date', 'anomaly_count']
    return timeline

def get_anomalies_per_stock(anomaly_df):
    """Count anomalies per ticker."""
    return (anomaly_df[anomaly_df['is_anomaly'] == 1]
            .groupby('ticker').size()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={0: 'count', 'ticker': 'ticker', 0: 'anomaly_count'}))
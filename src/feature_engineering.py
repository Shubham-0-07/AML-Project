import numpy as np

def add_features(df):
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['log_amount'] = df['amount'].apply(lambda x: np.log1p(x) if x > 0 else 0)
    return df

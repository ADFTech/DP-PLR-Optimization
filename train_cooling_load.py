#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import Dict, Any
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df['timestamp_15m'])
    df = df.copy()
    df['year'] = ts.dt.year
    df['month'] = ts.dt.month
    df['day'] = ts.dt.day
    df['hour'] = ts.dt.hour
    df['minute'] = ts.dt.minute
    df['weekday'] = ts.dt.weekday
    df['holiday'] = df['weekday'].isin([5, 6]).astype(int)
    return df

def train_cooling_load(df: pd.DataFrame, model_path: str = 'models/hlx_gbt_cooling_load_model.pkl', random_state: int = 0) -> Dict[str, Any]:
    required_cols = [
        'timestamp_15m', 'temp', 'humidity',
        'HLX_L18_AHU_Header_Low_Zone_Differential_Pressure',
        'HLX_B1_Chiller_BTU_METER_Cooling_Capacity',
        'HLX_B1_Chiller_DPM_MSB_IN_1_kW',
        'HLX_B1_Chiller_DPM_MSB_IN_2_kW'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError('Missing required columns: {m}'.format(m=missing))

    df = df[np.isfinite(df['HLX_B1_Chiller_BTU_METER_Cooling_Capacity']) & (df['HLX_B1_Chiller_BTU_METER_Cooling_Capacity'] != 0)]
    df = df.copy()
    df['kw_total'] = df['HLX_B1_Chiller_DPM_MSB_IN_1_kW'] + df['HLX_B1_Chiller_DPM_MSB_IN_2_kW']
    df = _add_time_features(df)


    # Map 'temp' -> 'temperature' for Model-A schema compatibility
    if 'temperature' not in df.columns and 'temp' in df.columns:
        df['temperature'] = df['temp']

    y = df['HLX_B1_Chiller_BTU_METER_Cooling_Capacity']

    features = [
        'temperature', 'humidity',
        'weekday', 'holiday',
        'year', 'month', 'day', 'hour', 'minute'
    ]
    X = df[features]

    n = len(df)
    split_idx = int(n * 0.9)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=300,
        tree_method='hist',
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.08,
        random_state=random_state,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred <= 150, 0, y_pred)

    r2 = float(r2_score(y_test, y_pred))
    rmse = float(mean_squared_error(y_test, y_pred, squared=False))

    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return {
        'target': 'HLX_B1_Chiller_BTU_METER_Cooling_Capacity',
        'rows': n,
        'r2': r2,
        'rmse': rmse,
        'model_path': model_path,
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Cooling Load model from CSV exported dataset.')
    parser.add_argument('--csv', required=True, help='Path to CSV from the view')
    parser.add_argument('--model-path', default='models/hlx_gbt_cooling_load_model.pkl')
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=['timestamp_15m'])
    metrics = train_cooling_load(df, model_path=args.model_path)
    print(metrics)
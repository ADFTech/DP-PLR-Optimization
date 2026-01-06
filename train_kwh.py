#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import Dict, Any
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

def train_kwh(df: pd.DataFrame, model_path: str = 'models/hlx_gbt_kwh_model.pkl', use_exact_kwh: bool = False, random_state: int = 0) -> Dict[str, Any]:
    # Map columns to old names. 
    col_map = [('temperature','temp'),
               ('HLX_B1_Chiller_BTU_METER_Cooling_Capacity_Header','HLX_B1_Chiller_BTU_METER_Cooling_Capacity'),
               ('Timestamp','timestamp_15m'),
               ('Year','year'),
               ('Month','month'),
               ('Day','day'),
               ('Hour','hour'),
               ('Minute','minute'),
               ]
    for newname,oldname in col_map:
        if newname not in df.columns and oldname in df.columns:
            df[newname] = df[oldname]
    

    # Guard: ensure at least one source column exists
    if 'kwh_15m_exact' not in df.columns and 'kwh_15m' not in df.columns:
        raise KeyError("Missing both 'kwh_15m_exact' and 'kwh_15m' columns.")

    # Create the two sources, coercing to numeric so NULLs are NaN
    exact = pd.to_numeric(df.get('kwh_15m_exact'), errors='coerce')  # may be None if col missing
    approx = pd.to_numeric(df.get('kwh_15m'), errors='coerce')       # may be None if col missing

    # Define “invalid exact” as NaN or 0.0 
    invalid_exact = exact.isna() | (exact == 0)

    # Fallback rule:
    # - Prefer exact where valid (not NaN and > 0)
    # - Otherwise use approx
    # - If approx is also missing, result will stay NaN
    df['KWH'] = np.where(~invalid_exact, exact, approx)  


    # if('KWH') not in df.columns:
    #     if use_exact_kwh and 'kwh_15m_exact' in df.columns:
    #         df['KWH'] = df['kwh_15m_exact']
    #     elif 'kwh_15m' in df.columns:
    #         df['KWH'] = df['kwh_15m']
    required_cols = [
        'Timestamp', 'temperature', 'humidity',
        'HLX_L18_AHU_Header_Low_Zone_Differential_Pressure',
        'HLX_B1_Chiller_BTU_METER_Cooling_Capacity_Header',
        'HLX_B1_Chiller_DPM_MSB_IN_1_kW',
        'HLX_B1_Chiller_DPM_MSB_IN_2_kW',
        'KWH'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError('Missing required columns: {m}'.format(m=missing))

    df = df.copy()
    df['kw_total'] = df['HLX_B1_Chiller_DPM_MSB_IN_1_kW'] + df['HLX_B1_Chiller_DPM_MSB_IN_2_kW']
    cap = df['HLX_B1_Chiller_BTU_METER_Cooling_Capacity_Header']
    df['kW_RT'] = np.where(cap == 0, 0, df['kw_total'] / cap)

    target_col = 'KWH'
    # # Coerce to numeric; non-convertible values become NaN
    # y_numeric = pd.to_numeric(df[target_col], errors='coerce')

    # # Mask of non-numeric (i.e., became NaN after coercion)
    # mask_non_numeric = y_numeric.isna() & df[target_col].notna()

    # print(f"Non-numeric label entries in '{target_col}':")
    # for idx, val in df.loc[mask_non_numeric, target_col].items():
    #     print(f"  idx={idx}  value={repr(val)}")
    # df = _add_time_features(df)

    # Ensure capitalized time features exist even if mapping didn't create them
    if not all(c in df.columns for c in ['Year','Month','Day','Hour','Minute']):
        # use Timestamp (already mapped if needed)
        ts = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['Year']   = ts.dt.year
        df['Month']  = ts.dt.month
        df['Day']    = ts.dt.day
        df['Hour']   = ts.dt.hour
        df['Minute'] = ts.dt.minute


    # Coerce numeric types for safety
    df['KWH'] = pd.to_numeric(df['KWH'], errors='coerce')
    df[['kW_RT','temperature','humidity','Year','Month','Day','Hour','Minute']] = \
        df[['kW_RT','temperature','humidity','Year','Month','Day','Hour','Minute']].apply(pd.to_numeric, errors='coerce')

    # # df = df[np.isfinite(df[target_col]) & (df[target_col] != 0)]

    
    features = [
        'HLX_L18_AHU_Header_Low_Zone_Differential_Pressure',
        'HLX_B1_Chiller_BTU_METER_Cooling_Capacity_Header',
        'kW_RT',
        'temperature', 'humidity',
        'Year', 'Month', 'Day', 'Hour', 'Minute'
    ]

    # Coerce numerics (already present for key columns—this is just belt-and-braces)
    X = df[features].apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(df['KWH'], errors='coerce')

    y = df[target_col]

    X = df[features]

    # Post-fit schema guard
    # (ensures the booster recorded the exact expected feature order)
    expected = [
        'HLX_L18_AHU_Header_Low_Zone_Differential_Pressure',
        'HLX_B1_Chiller_BTU_METER_Cooling_Capacity_Header',
        'kW_RT',
        'temperature','humidity',
        'Year','Month','Day','Hour','Minute'
    ]
    
    n = len(df)
    split_idx = int(n * 0.9)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=400,
        tree_method='hist',
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.06,
        random_state=random_state,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # after model.fit(...)
    assert list(model.feature_names_in_) == expected, \
        f"Feature order mismatch: {list(model.feature_names_in_)}"
        
    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(mean_squared_error(y_test, y_pred, squared=False))

    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return {
        'target': target_col,
        'rows': n,
        'r2': r2,
        'rmse': rmse,
        'model_path': model_path,
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train KWH model from CSV exported dataset.')
    parser.add_argument('--csv', required=True, help='Path to CSV with columns from the view')
    parser.add_argument('--model-path', default='models/hlx_gbt_kwh_model.pkl')
    parser.add_argument('--use-exact-kwh', action='store_true', help='Use kwh_15m_exact as target if present')
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=['timestamp_15m'])
    metrics = train_kwh(df, model_path=args.model_path, use_exact_kwh=args.use_exact_kwh)
    print(metrics)
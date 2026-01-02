import os
import pickle as p
import pandas as pd
import numpy as np
from math import sqrt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import random

class CoolingLoadTrainer:
    """
    Train an XGBoost model to predict cooling load.
    """

    def __init__(self, csv_path: str, model_path: str = "Model/hlx_gbt_cl_model.p") -> None:
        self.csv_path = csv_path
        self.model_path = model_path

    # ------------------------------------------------------------------
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate time-based features from timestamp column.
        """
        timestamp_col = None
        random.seed(10)
      
        for col in df.columns:
            if 'timestamp' in col.lower() or 'date' in col.lower():
                timestamp_col = col
                break

        if not timestamp_col:
            raise KeyError("No timestamp column found in dataset")

        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df['year'] = df[timestamp_col].dt.year
        df['month'] = df[timestamp_col].dt.month
        df['day'] = df[timestamp_col].dt.day
        df['hour'] = df[timestamp_col].dt.hour
        df['minute'] = df[timestamp_col].dt.minute
        df['weekday'] = df[timestamp_col].dt.weekday
        df['holiday'] = df['weekday'].isin([5, 6]).astype(int)  # weekend = holiday

        df.attrs['timestamp_col'] = timestamp_col  # store detected column name
        return df

    # ------------------------------------------------------------------
    def train(self) -> None:
        """
        Train the cooling load model.
        """
        print(f"[INFO] Reading dataset from: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        df = self._create_time_features(df)

        if 'temperature' not in df.columns:
            raise KeyError("Missing column: 'temperature' in dataset")
        if 'HLX_B1_Chiller_BTU_METER_Cooling_Capacity' not in df.columns:
            raise KeyError("Missing column: 'HLX_B1_Chiller_BTU_METER_Cooling_Capacity' in dataset")

        features = [
            'temperature','humidity', 'weekday', 'holiday', 'year',
            'month', 'day', 'hour', 'minute'
        ]

        X = df[features]
        y = df['HLX_B1_Chiller_BTU_METER_Cooling_Capacity']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=0
        )
        print("[INFO] Training XGBoost model...")
        model = XGBRegressor(n_estimators=100, seed=0)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred = [0 if val <= 150 else val for val in y_pred]

        r2 = r2_score(y_test, y_pred)
        rmse = sqrt(mean_squared_error(y_test, y_pred))

        print(f"[RESULT] R² Score: {r2:.3f}")
        print(f"[RESULT] RMSE: {rmse:.3f}")

        # Save model
        with open(self.model_path, 'wb') as f:
            p.dump(model, f)
        print(f"[INFO] Model saved to: {self.model_path}")

    # ------------------------------------------------------------------
    def predict(self, new_data_csv: str) -> pd.DataFrame:
        """
        Predict cooling load for new CSV input.
        Apply rule: if predicted <= 150, set to zero.
        Fill short zero periods (≤1 hour) between 8am-6pm using bfill → ffill.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Trained model not found. Please train first.")

        print(f"[INFO] Loading model from {self.model_path}")
        with open(self.model_path, 'rb') as f:
            model = p.load(f)

        new_data = pd.read_csv(new_data_csv)
        new_data = self._create_time_features(new_data)

        timestamp_col = new_data.attrs.get('timestamp_col', None)
        if not timestamp_col:
            raise KeyError("No timestamp column found in new data.")

        if 'temperature' not in new_data.columns:
            raise KeyError("Missing 'temperature' column in new data.")

        features = [
            'temperature','humidity', 'weekday', 'holiday', 'year',
            'month', 'day', 'hour', 'minute'
        ]

        preds = model.predict(new_data[features])
        preds = [0 if val <= 150 else val for val in preds]

        result = pd.DataFrame({
            "timestamp": pd.to_datetime(new_data[timestamp_col]),
            "predicted_RT": preds
        })

        # ------------------------------
        # Fill short zero gaps (≤1 hour) between 08:00-18:00
        # ------------------------------
        MAX_FILL_STEPS = 6  # adjust based on your interval (e.g., 10-min = 6 steps)
        mask_working = result["timestamp"].dt.hour.between(8, 17)
        mask_zero = result["predicted_RT"] == 0
        target_mask = mask_working & mask_zero

        i = 0
        n = len(result)
        while i < n:
            if target_mask.iloc[i]:
                start = i
                while i < n and target_mask.iloc[i]:
                    i += 1
                end = i
                gap_len = end - start
                if gap_len <= MAX_FILL_STEPS:
                    result.loc[start:end-1, "predicted_RT"] = (
                        result["predicted_RT"]
                        .replace(0, np.nan)
                        .bfill()
                        .ffill()
                        .iloc[start:end]
                    )
            else:
                i += 1

        print(f"[INFO] Prediction complete: {len(result)} records processed.")
        return result


# ----------------------------------------------------------------------
if __name__ == "__main__":
    trainer = CoolingLoadTrainer(csv_path="chiller_train.csv")

    # 1️⃣ Train the model
    trainer.train()

    # 2️⃣ Predict on new data
    pred_df = trainer.predict("chiller_test.csv")

    # 3️⃣ Save predictions
    pred_df.to_csv("predicted_RT_hlx.csv", index=False)
    print("[INFO] Predicted results saved to predicted_RT_hlx.csv")

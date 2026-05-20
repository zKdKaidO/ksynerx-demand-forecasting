import os
import yaml
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- REQUEST SCHEMA (ĐÃ BỎ CỘT DATE) ---
class ForecastInput(BaseModel):
    unique_id: str
    selling_price: float
    is_promotion: int
    holiday: int
    precipitation: float
    temperature: float
    humidity: float
    wind_level: float

# --- API INITIALIZATION ---
app = FastAPI(
    title="Fresh Retail Next-Day Forecast API",
    description="Simulate tomorrow's demand based on user-provided weather and business conditions.",
    version="2.0.0"
)

def load_model_pipeline():
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    return joblib.load(cfg['model']['model_export_path'])

try:
    pipeline = load_model_pipeline()
    print("[SUCCESS] Model loaded into API state.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    pipeline = None

@app.post("/predict_next_day")
def predict_next_day_demand(payload: ForecastInput):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model unavailable.")

    try:
        # 1. Hệ thống TỰ ĐỘNG sinh ra khung thời gian của "Ngày mai"
        X_df = pipeline.make_future_dataframe(h=1)

        if payload.unique_id not in X_df['unique_id'].values:
            raise HTTPException(status_code=404, detail=f"ID '{payload.unique_id}' not found.")

        # 2. Xây dựng các biến Động từ thời gian tự động và kịch bản người dùng
        X_df['day_of_week'] = X_df['ds'].dt.dayofweek
        X_df['month'] = X_df['ds'].dt.month
        X_df['is_weekend'] = X_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        X_df['is_month_start'] = X_df['ds'].dt.is_month_start.astype(int)
        X_df['is_month_end'] = X_df['ds'].dt.is_month_end.astype(int)

        X_df['holiday'] = payload.holiday
        X_df['is_promotion'] = payload.is_promotion
        X_df['selling_price'] = payload.selling_price
        X_df['precipitation'] = payload.precipitation
        X_df['temperature'] = payload.temperature
        X_df['humidity'] = payload.humidity
        X_df['wind_level'] = payload.wind_level
        X_df['is_raining'] = 1 if payload.precipitation > 0 else 0
        X_df['is_hot_weather'] = 1 if payload.temperature > 32 else 0

        # 3. Ép kiểu Category
        cat_cols = [
            'day_of_week', 'month', 'is_weekend', 'is_month_start',
            'is_month_end', 'holiday', 'is_promotion', 'is_raining', 'is_hot_weather'
        ]
        for col in cat_cols:
            X_df[col] = X_df[col].astype('category')

        # 4. Loại bỏ biến tĩnh
        static_features = [
            'store_id', 'product_id', 'location', 
            'management_group', 'category_1', 'category_2', 'category_3'
        ]
        X_df_pred = X_df.drop(columns=static_features, errors='ignore')

        # 5. Dự báo
        predictions = pipeline.predict(h=1, X_df=X_df_pred)

        # 6. Trích xuất kết quả
        target_pred = predictions[predictions['unique_id'] == payload.unique_id].iloc[0]
        
        return {
            "unique_id": payload.unique_id,
            "forecast_date": target_pred['ds'].strftime('%Y-%m-%d'),
            "predicted_adjusted_demand": max(0.0, round(float(target_pred['lightgbm']), 2))
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
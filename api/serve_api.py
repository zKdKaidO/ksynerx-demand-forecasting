import os
import yaml
import joblib
import pandas as pd
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# --- BẬT HỆ THỐNG LOGGING CHUẨN PRODUCTION (Fix Issue 8) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api_serving")

# --- REQUEST SCHEMA VỚI RÀO CHẮN NGHIÊM NGẶT (Fix Issue 7) ---
class ForecastInput(BaseModel):
    unique_id: str = Field(..., description="Store and product ID combination")
    selling_price: float = Field(..., ge=0.0, description="Selling price cannot be negative")
    is_promotion: int = Field(..., ge=0, le=1, description="Binary flag: 0 or 1")
    holiday: int = Field(..., ge=0, le=1, description="Binary flag: 0 or 1")
    precipitation: float = Field(..., ge=0.0, description="Rainfall in mm")
    temperature: float = Field(..., ge=-50.0, le=60.0, description="Temperature in Celsius")
    humidity: float = Field(..., ge=0.0, le=100.0, description="Humidity percentage (0-100)")
    wind_level: float = Field(..., ge=0.0, description="Wind level >= 0")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "unique_id": "120_112",
                "selling_price": 1.0,
                "is_promotion": 0,
                "holiday": 0,
                "precipitation": 0.0,
                "temperature": 28.5,
                "humidity": 70.0,
                "wind_level": 1.5
            }]
        }
    }

# --- MODEL LOADING LOGIC ---
def load_app_config():
    BASE_DIR = Path(__file__).resolve().parent.parent
    config_path = BASE_DIR / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file missing at {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f), BASE_DIR

def load_model_pipeline(cfg, base_dir):
    model_abs_path = base_dir / cfg['model']['model_export_path']
    if not model_abs_path.exists():
        raise FileNotFoundError(f"Model file missing at {model_abs_path}")
    return joblib.load(model_abs_path)

# --- APP LIFESPAN MANAGEMENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing API and loading configs...")
    try:
        cfg, base_dir = load_app_config()
        app.state.cfg = cfg
        app.state.pipeline = load_model_pipeline(cfg, base_dir)
        
        logger.info("Pre-computing future dataframe (Cache)...")
        app.state.base_X_df = app.state.pipeline.make_future_dataframe(h=1)
        logger.info("API successfully started and ready.")
    except Exception as e:
        logger.error(f"Critical Startup Failure: {e}", exc_info=True)
        # Nếu model hỏng, cho API sập luôn (Fail-Fast)
        raise e
        
    yield 
    
    logger.info("Shutting down API and releasing resources...")
    app.state.pipeline = None
    app.state.base_X_df = None

# --- API INITIALIZATION ---
app = FastAPI(
    title="Fresh Retail Demand API",
    description="Enterprise-grade API with strict validation, observability, and dynamic configurations.",
    version="4.0.0",
    lifespan=lifespan
)

# --- ENDPOINTS ---
@app.post("/predict_next_day")
def predict_next_day_demand(payload: ForecastInput, request: Request):
    pipeline = request.app.state.pipeline
    base_X_df = request.app.state.base_X_df
    cfg = request.app.state.cfg
    
    if pipeline is None or base_X_df is None:
        logger.error("Predict endpoint called but model/cache is uninitialized.")
        raise HTTPException(status_code=500, detail="Internal Service Error: Model unavailable.")

    try:
        # Lọc $O(1)$ ngay lập tức
        X_df_full = base_X_df.copy()
        if payload.unique_id not in X_df_full['unique_id'].values:
            raise ValueError(f"ID '{payload.unique_id}' not found in training history.")

        X_df = X_df_full[X_df_full['unique_id'] == payload.unique_id].copy()

        # Feature Engineering
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

        # Ép kiểu
        cat_cols = ['day_of_week', 'month', 'is_weekend', 'is_month_start', 'is_month_end', 'holiday', 'is_promotion', 'is_raining', 'is_hot_weather']
        for col in cat_cols:
            X_df[col] = X_df[col].astype('category')

        # --- FIX ISSUE 5: Lấy static features từ file config thay vì hardcode ---
        static_features = cfg.get('data', {}).get('static_features', [])
        # Chỉ drop những cột thực sự tồn tại, bỏ errors='ignore' rủi ro
        cols_to_drop = [c for c in static_features if c in X_df.columns]
        X_df_pred = X_df.drop(columns=cols_to_drop)

        # Chạy dự báo
        predictions = pipeline.predict(h=1, X_df=X_df_pred)
        target_pred = predictions.iloc[0]

        # --- FIX ISSUE 6: Tự động dò tìm tên cột dự đoán ---
        # mlforecast luôn trả về unique_id, ds, và [tên_model]
        model_cols = [col for col in predictions.columns if col not in ['unique_id', 'ds']]
        if not model_cols:
            raise RuntimeError("Model prediction column missing from output.")
        model_name = model_cols[0] 

        predicted_value = max(0.0, round(float(target_pred[model_name]), 2))
        
        logger.info(f"Successful prediction for {payload.unique_id}: {predicted_value}")
        return {
            "unique_id": payload.unique_id,
            "forecast_date": target_pred['ds'].strftime('%Y-%m-%d'),
            "model_engine": model_name,
            "predicted_adjusted_demand": predicted_value
        }

    # --- FIX ISSUE 9: Phân loại rạch ròi 400 vs 500 ---
    except ValueError as ve:
        # Lỗi Client (400): Do user gửi ID vớ vẩn
        logger.warning(f"Bad Request: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Lỗi Server (500): Tràn RAM, lỗi Pandas, lỗi hàm số...
        logger.error(f"Internal Server Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")
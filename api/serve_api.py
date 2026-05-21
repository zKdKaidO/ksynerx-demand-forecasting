import os
import yaml
import joblib
import pandas as pd
import logging
import uuid
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# --- LOGGING VỚI CORRELATION ID (Fix Issue 1.7) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api_serving")

# --- API AUTHENTICATION (Fix Issue 1.8) ---
API_KEY = "ksynerx-secret-key-2026"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# --- REQUEST SCHEMA ---
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
        raise e
        
    yield 
    
    logger.info("Shutting down API and releasing resources...")
    app.state.pipeline = None
    app.state.base_X_df = None

# --- API INITIALIZATION ---
app = FastAPI(
    title="Fresh Retail Demand API",
    description="Enterprise-grade API with auth, async inference, and real-time cache injection.",
    version="5.0.0",
    lifespan=lifespan
)

# --- MIDDLEWARE: THEO DÕI REQUEST TỪNG NGƯỜI DÙNG (Fix Issue 1.7) ---
@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response

# --- HEALTH ENDPOINT (Fix Issue 1.4) ---
@app.get("/health")
async def health_check(request: Request):
    if getattr(request.app.state, 'pipeline', None) is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model not loaded")
    return {"status": "ok", "message": "Service is healthy and fully operational."}

# --- PREDICT ENDPOINT BẤT ĐỒNG BỘ (Fix Issue 1.3) ---
@app.post("/predict_next_day")
async def predict_next_day_demand(
    payload: ForecastInput, 
    request: Request, 
    api_key: str = Depends(verify_api_key)  # Yêu cầu nhập Key
):
    pipeline = request.app.state.pipeline
    base_X_df = request.app.state.base_X_df
    cfg = request.app.state.cfg
    req_id = request.state.request_id
    
    if pipeline is None or base_X_df is None:
        logger.error(f"[{req_id}] Predict endpoint called but model/cache is uninitialized.")
        raise HTTPException(status_code=503, detail="Internal Service Error: Model unavailable.")

    try:
        # --- FIX ISSUE 1.1: Lọc đúng 1 dòng TRƯỚC KHI copy (O(1) Memory) ---
        row = base_X_df[base_X_df['unique_id'] == payload.unique_id]
        if row.empty:
            raise ValueError(f"ID '{payload.unique_id}' not found in training history.")
        X_df = row.copy()

        # --- FIX ISSUE 1.2: Chống Cache thiu, ép ngày dự báo thành 'Ngày mai' ở thời điểm hiện tại ---
        actual_tomorrow = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
        X_df['ds'] = actual_tomorrow

        # Feature Engineering
        X_df['day_of_week'] = X_df['ds'].dt.dayofweek
        X_df['month'] = X_df['ds'].dt.month
        
        # --- FIX ISSUE 1.5 (CRITICAL): Bổ sung cột năm và tuần để khớp với Train ---
        X_df['week_of_year'] = X_df['ds'].dt.isocalendar().week.astype(int)
        X_df['year'] = X_df['ds'].dt.year
        
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
        cat_cols = ['day_of_week', 'month', 'week_of_year', 'year', 'is_weekend', 'is_month_start', 'is_month_end', 'holiday', 'is_promotion', 'is_raining', 'is_hot_weather']
        for col in cat_cols:
            X_df[col] = X_df[col].astype('category')

        # Drop tính năng tĩnh (Chống lỗi thiếu tên cột - Issue 1.6)
        static_features = cfg.get('data', {}).get('static_features', [])
        cols_to_drop = [c for c in static_features if c in X_df.columns]
        X_df_pred = X_df.drop(columns=cols_to_drop)

        # Chạy model bằng Thread Pool chống sập sự kiện vòng lặp (Fix Issue 1.3)
        loop = asyncio.get_running_loop()
        predictions = await loop.run_in_executor(None, lambda: pipeline.predict(h=1, X_df=X_df_pred))
        target_pred = predictions.iloc[0]

        # --- FIX ISSUE 3.8: Tự đọc tên Model chạy chính từ Config ---
        model_name = cfg.get('model', {}).get('algorithm', 'lightgbm')
        if model_name not in target_pred:
            raise RuntimeError(f"Configured model '{model_name}' not found in prediction output.")

        predicted_value = max(0.0, round(float(target_pred[model_name]), 2))
        
        logger.info(f"[{req_id}] Successful prediction for {payload.unique_id}: {predicted_value}")
        return {
            "unique_id": payload.unique_id,
            "forecast_date": target_pred['ds'].strftime('%Y-%m-%d'),
            "model_engine": model_name,
            "predicted_adjusted_demand": predicted_value,
            "request_id": req_id
        }

    except ValueError as ve:
        logger.warning(f"[{req_id}] Bad Request: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"[{req_id}] Internal Server Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")
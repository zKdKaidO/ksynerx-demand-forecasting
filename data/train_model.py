import os
import pandas as pd
import yaml
import joblib
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_config(config_file: str = "config/config.yaml") -> dict:
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def format_data(df: pd.DataFrame) -> pd.DataFrame:
    """Format columns to meet MLForecast requirements and prevent data leakage."""
    processed_df = df.rename(columns={'adjusted_demand': 'y'})
    
    # Drop leaky columns
    leaky_cols = ['units_ordered', 'stockout_flag']
    processed_df = processed_df.drop(columns=leaky_cols, errors='ignore')
    
    # Enforce Category type for LightGBM
    cat_cols = [
        'store_id', 'product_id', 'location', 
        'management_group', 'category_1', 'category_2', 'category_3',
        'day_of_week', 'month', 'is_weekend', 'is_month_start', 
        'is_month_end', 'holiday', 'is_promotion', 'is_raining', 'is_hot_weather'
    ]
    for col in cat_cols:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype('category')
            
    return processed_df

def build_and_train_pipeline(train_df: pd.DataFrame, model_cfg: dict) -> MLForecast:
    """Initialize LightGBM, wrap in MLForecast, and fit the data."""
    print("[INFO] Initializing LightGBM and MLForecast pipeline...")
    
    model = LGBMRegressor(**model_cfg['lgbm_params'])
    
    fcst = MLForecast(
        models={'lightgbm': model},
        freq='D', 
        lags=model_cfg['lags'],
        # CÚ PHÁP MỚI ĐÃ ĐƯỢC SỬA Ở DÒNG DƯỚI ĐÂY:
        lag_transforms={1: [RollingMean(window_size=7)]}
    )
    
    static_features = [
        'store_id', 'product_id', 'location', 
        'management_group', 'category_1', 'category_2', 'category_3'
    ]
    
    print("[INFO] Training model (fitting on Train Set)...")
    fcst.fit(train_df, static_features=static_features)
    return fcst

def evaluate_model(fcst: MLForecast, eval_df: pd.DataFrame):
    """Predict on eval set and calculate error metrics."""
    print("[INFO] Running predictions on Eval Set...")
    
    horizon = eval_df['ds'].nunique()
    
    # 1. Khai báo lại danh sách biến tĩnh đã dùng lúc train
    static_features = [
        'store_id', 'product_id', 'location', 
        'management_group', 'category_1', 'category_2', 'category_3'
    ]
    
    # 2. X_df = Bỏ cột mục tiêu 'y' VÀ bỏ luôn các biến tĩnh
    X_df = eval_df.drop(columns=['y'] + static_features)
    
    # 3. Dự báo
    predictions = fcst.predict(h=horizon, X_df=X_df)
    
    # Merge predictions with actual true values (y) to compare
    results = predictions.merge(eval_df[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='inner')
    
    mae = mean_absolute_error(results['y'], results['lightgbm'])
    rmse = mean_squared_error(results['y'], results['lightgbm']) ** 0.5
    
    print("\n" + "="*40)
    print("        MODEL EVALUATION RESULTS        ")
    print("="*40)
    print(f" Mean Absolute Error (MAE) : {mae:.4f}")
    print(f" Root Mean Squared Error   : {rmse:.4f}")
    print("="*40 + "\n")

def main():
    cfg = load_config()
    data_cfg = cfg['data']['files']
    model_cfg = cfg['model']
    
    print("[INFO] Loading Train and Eval datasets...")
    raw_train = pd.read_parquet(data_cfg['train']['processed'])
    raw_eval = pd.read_parquet(data_cfg['eval']['processed'])
    
    train_df = format_data(raw_train)
    eval_df = format_data(raw_eval)
    
    fcst_pipeline = build_and_train_pipeline(train_df, model_cfg)
    
    evaluate_model(fcst_pipeline, eval_df)
    
    export_path = model_cfg['model_export_path']
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    joblib.dump(fcst_pipeline, export_path)
    print(f"[SUCCESS] Trained model saved to: {export_path}")

if __name__ == "__main__":
    main()
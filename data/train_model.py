from pathlib import Path
import pandas as pd
import yaml
import joblib
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from window_ops.rolling import rolling_mean, rolling_std

# --- FIX ISSUE 2.3.7: THÊM CÁC MÔ HÌNH SO SÁNH ---
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_config():
    BASE_DIR = Path(__file__).resolve().parent.parent
    config_path = BASE_DIR / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"[CRITICAL] Can't find config at: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    
    print("[INFO] Loading datasets...")
    train_df = pd.read_parquet(cfg['data']['files']['train']['processed'])
    eval_df = pd.read_parquet(cfg['data']['files']['eval']['processed'])
    
    # Chuyển đổi Category
    static_features = cfg['data'].get('static_features', [])
    cat_cols = ['day_of_week', 'month', 'week_of_year', 'year', 'is_weekend', 'is_month_start', 'is_month_end']
    for col in static_features + cat_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype('category')
            eval_df[col] = eval_df[col].astype('category')

    object_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    if 'unique_id' in object_cols:
        object_cols.remove('unique_id')
        
    if object_cols:
        print(f"[WARNING] Dropping text/object columns to prevent LightGBM crash: {object_cols}")
        train_df = train_df.drop(columns=object_cols)
        eval_df = eval_df.drop(columns=object_cols, errors='ignore')
    # Khởi tạo mô hình
    lgbm_params = cfg['model']['lgbm_params']
    
    # Đưa nhiều mô hình vào thi đấu cùng lúc
    models = {
        'lightgbm': LGBMRegressor(**lgbm_params),
        'ridge_baseline': Ridge(alpha=1.0),
        'random_forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    }
    
    # --- FIX ISSUE 2.3.1: Đa dạng hóa Lag Transforms ---
    fcst_pipeline = MLForecast(
        models=models,
        freq='D',
        lags=cfg['model']['lags'],
        lag_transforms={
            1: [(rolling_mean, 7), (rolling_std, 7)],  # Trung bình và độ lệch chuẩn 1 tuần qua
            7: [(rolling_mean, 14)]                    # Trung bình 2 tuần dựa trên độ trễ 1 tuần
        },
        date_features=['dayofweek', 'month'],
        num_threads=4
    )
    
    print("[INFO] Training Multi-Model Pipeline (This may take a minute)...")
    fcst_pipeline.fit(
        train_df, 
        id_col='unique_id', 
        time_col='ds', 
        target_col='y', 
        static_features=static_features
    )
    
    # Đánh giá đa mô hình
    print("[INFO] Running predictions on Eval Set...")
    horizon = eval_df['ds'].nunique()
    
    # Bỏ target và static features khi dự báo
    X_df = eval_df.drop(columns=['y'] + static_features, errors='ignore')
    predictions = fcst_pipeline.predict(h=horizon, X_df=X_df)
    
    results = predictions.merge(eval_df[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='inner')
    
    print("\n" + "="*50)
    print("      MULTI-MODEL EVALUATION RESULTS      ")
    print("="*50)
    for model_name in models.keys():
        mae = mean_absolute_error(results['y'], results[model_name])
        rmse = mean_squared_error(results['y'], results[model_name]) ** 0.5
        print(f"[{model_name.upper()}]")
        print(f"  MAE  : {mae:.4f}")
        print(f"  RMSE : {rmse:.4f}")
        print("-" * 50)
        
    joblib.dump(fcst_pipeline, cfg['model']['model_export_path'])
    print(f"[SUCCESS] Trained multi-model pipeline saved to: {cfg['model']['model_export_path']}")

if __name__ == "__main__":
    main()
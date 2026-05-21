import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from mlforecast import MLForecast
from window_ops.rolling import rolling_mean, rolling_std

from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_config():
    """Load config bằng đường dẫn tuyệt đối (Fix Issue 4.2)"""
    BASE_DIR = Path(__file__).resolve().parent.parent
    config_path = BASE_DIR / "config" / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"[CRITICAL] Không tìm thấy config tại: {config_path}")
        
    with open(config_path, "r") as f:
        return yaml.safe_load(f), BASE_DIR

def calculate_wmape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Tính toán wMAPE để đánh giá công bằng cho hàng tươi sống (Fix Issue 3.7)"""
    sum_abs_error = np.sum(np.abs(y_true - y_pred))
    sum_actual = np.sum(np.abs(y_true))
    if sum_actual == 0:
        return 0.0
    return sum_abs_error / sum_actual

def main():
    cfg, BASE_DIR = load_config()
    
    print("[INFO] Loading datasets...")
    train_df = pd.read_parquet(BASE_DIR / cfg['data']['files']['train']['processed'])
    eval_df = pd.read_parquet(BASE_DIR / cfg['data']['files']['eval']['processed'])
    
    # Lọc Static Features thực tế
    raw_static_features = cfg['data'].get('static_features', [])
    static_features = [col for col in raw_static_features if col in train_df.columns]

    # Chuyển đổi Category
    cat_cols = ['day_of_week', 'month', 'week_of_year', 'year', 'is_weekend', 'is_month_start', 'is_month_end']
    for col in static_features + cat_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype('category')
            eval_df[col] = eval_df[col].astype('category')

    # Dọn dẹp cột rác (Object/Text)
    object_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    if 'unique_id' in object_cols:
        object_cols.remove('unique_id')
    if object_cols:
        train_df = train_df.drop(columns=object_cols)
        eval_df = eval_df.drop(columns=object_cols, errors='ignore')

    # --- FIX ISSUE 3.4 & 3.5: Cấu hình mô hình mạnh mẽ và công bằng hơn ---
    lgbm_params = cfg['model']['lgbm_params']
    # Bổ sung các tham số chống Overfit cho LightGBM
    lgbm_params.update({'num_leaves': 64, 'min_child_samples': 40})
    
    models = {
        'lightgbm': LGBMRegressor(**lgbm_params),
        'ridge_baseline': Ridge(alpha=1.0),
        # Nâng n_estimators lên 200 để Random Forest đủ sức cạnh tranh
        'random_forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    }
    
    # --- FIX ISSUE 3.1 & 3.2 & 3.3: Dọn dẹp MLForecast pipeline ---
    fcst_pipeline = MLForecast(
        models=models,
        freq='D',
        lags=cfg['model']['lags'],
        lag_transforms={
            1: [(rolling_mean, 7), (rolling_std, 7)],  
            7: [(rolling_mean, 14), (rolling_std, 14)] # Bổ sung Std cho lag_7
        },
        # XÓA date_features vì đã tự làm ở khâu Data Prep rồi
        num_threads=4
    )
    
    print("[INFO] Training Multi-Model Pipeline...")
    fcst_pipeline.fit(
        train_df, 
        id_col='unique_id', 
        time_col='ds', 
        target_col='y', 
        static_features=static_features
    )
    
    print("[INFO] Running predictions on Eval Set...")
    horizon = eval_df['ds'].nunique()
    
    X_df = eval_df.drop(columns=['y'] + static_features, errors='ignore')
    predictions = fcst_pipeline.predict(h=horizon, X_df=X_df)
    
    results = predictions.merge(eval_df[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='inner')
    
    # --- FIX ISSUE 3.6: Đánh giá theo từng bước thời gian (Horizon) ---
    results['horizon_step'] = results.groupby('unique_id').cumcount() + 1
    
    print("\n" + "="*60)
    print("      MULTI-MODEL EVALUATION RESULTS (SRE STANDARD)      ")
    print("="*60)
    
    for model_name in models.keys():
        mae = mean_absolute_error(results['y'], results[model_name])
        rmse = mean_squared_error(results['y'], results[model_name]) ** 0.5
        wmape_score = calculate_wmape(results['y'], results[model_name])
        
        print(f"\n🚀 [{model_name.upper()}] GLOBAL METRICS:")
        print(f"   MAE   : {mae:.4f}")
        print(f"   RMSE  : {rmse:.4f}")
        print(f"   wMAPE : {wmape_score:.2%}")
        
        # In MAE của 3 ngày dự báo đầu tiên để xem mô hình có bị "xuống sức" không
        print(f"   --- Horizon Breakdown (First 3 Days) ---")
        for h in [1, 2, 3]:
            h_data = results[results['horizon_step'] == h]
            if not h_data.empty:
                h_mae = mean_absolute_error(h_data['y'], h_data[model_name])
                print(f"   Day {h} MAE: {h_mae:.4f}")
    
    # Lưu file với đường dẫn tuyệt đối an toàn
    export_path = BASE_DIR / cfg['model']['model_export_path']
    export_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fcst_pipeline, export_path)
    print(f"\n[SUCCESS] Trained pipeline safely saved to: {export_path}")

if __name__ == "__main__":
    main()
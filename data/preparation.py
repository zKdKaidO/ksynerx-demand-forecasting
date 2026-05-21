import pandas as pd
import numpy as np
import yaml
from pathlib import Path

def load_config():
    BASE_DIR = Path(__file__).resolve().parent.parent
    config_path = BASE_DIR / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"[CRITICAL] No config file found at: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def filter_target_city(df: pd.DataFrame, target_city: int) -> pd.DataFrame:
    print(f"[INFO] Filtering data for target_city: {target_city}")
    filtered = df[df['city_id'] == target_city].copy()
    if filtered.empty:
        print(f"[WARNING] No data found for city_id {target_city}")
    return filtered

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()
    
    # thời gian
    dates = pd.to_datetime(processed['dt'])
    processed['ds'] = dates
    processed['week_of_year'] = dates.dt.isocalendar().week.astype(int)
    processed['year'] = dates.dt.year
    
    # Unique ID
    processed['unique_id'] = processed['store_id'].astype(str) + "_" + processed['product_id'].astype(str)
    
    # Adjusted Demand (LDR)
    available_hours = 16 - processed['stock_hour6_22_cnt']
    processed['adjusted_demand'] = np.where(
        processed['sale_amount'] == 0, 
        0.0, 
        processed['sale_amount'] * (16 / np.maximum(available_hours, 1))
    )
    
    # Log và Cap Outliers (Giới hạn giá trị ngoại lai)
    p99 = processed['adjusted_demand'].quantile(0.99)
    print(f"[INFO] 99th percentile of adjusted_demand is: {p99:.2f}. Capping extreme outliers.")
    # Cắt ngọn (clip) ở mức phân vị 99.9% để tránh các ca nhu cầu bùng nổ vô lý (24x, 50x)
    p999 = processed['adjusted_demand'].quantile(0.999)
    processed['adjusted_demand'] = processed['adjusted_demand'].clip(upper=p999)
    
    # Xử lý các cột tên gốc
    processed['selling_price'] = processed['discount']
    processed['y'] = processed['adjusted_demand']
    
    return processed

def check_date_gaps(df: pd.DataFrame):
    """Date Gaps"""
    temp = df.sort_values(['unique_id', 'ds'])
    temp['date_diff'] = temp.groupby('unique_id')['ds'].diff().dt.days
    gaps = temp[temp['date_diff'] > 1]
    
    if not gaps.empty:
        num_gaps = gaps['unique_id'].nunique()
        print(f"[WARNING] {num_gaps} series have date gaps > 1 day. This may distort lag features!")
    else:
        print("[INFO] No date gaps detected. Time series is contiguous.")

def validate_data(df: pd.DataFrame):
    print("[INFO] Running Data Validation...")
    if (df['sale_amount'] < 0).any():
        raise ValueError("Critical Data Error: Found negative sales amounts.")
    if df.duplicated(subset=['unique_id', 'ds']).any():
        print("[WARNING] Found duplicate (store, product, date) records. Dropping duplicates.")
        df = df.drop_duplicates(subset=['unique_id', 'ds'])
    
    check_date_gaps(df)
    print("[SUCCESS] Data validation passed.")
    return df

def main():
    cfg = load_config()
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    print("[INFO] Loading raw Train and Eval data...")
    train_df = pd.read_parquet(BASE_DIR / cfg['data']['files']['train']['raw'])
    eval_df = pd.read_parquet(BASE_DIR / cfg['data']['files']['eval']['raw'])
    
    # Áp dụng lọc thành phố nếu có trong config
    target_city = cfg['data'].get('target_city')
    if target_city is not None:
        train_df = filter_target_city(train_df, target_city)
        eval_df = filter_target_city(eval_df, target_city)
    
    print("[INFO] Processing Train data...")
    train_ready = build_features(train_df)
    train_ready = validate_data(train_ready)
    
    print("[INFO] Processing Eval data...")
    eval_ready = build_features(eval_df)
    eval_ready = validate_data(eval_ready)
    
    # --- (Leakage) ---
    # Giữ lại 'y', 'ds', 'unique_id', 'selling_price' và các cột static/features. 
    # Xóa hoàn toàn các cột trung gian tạo ra 'y'.
    leakage_cols = ['dt', 'adjusted_demand', 'sale_amount', 'stock_hour6_22_cnt', 'discount']
    drop_cols = [c for c in leakage_cols if c in train_ready.columns]
    
    train_ready = train_ready.drop(columns=drop_cols)
    eval_ready = eval_ready.drop(columns=drop_cols)
    
    # Lưu file
    train_out_path = BASE_DIR / cfg['data']['files']['train']['processed']
    eval_out_path = BASE_DIR / cfg['data']['files']['eval']['processed']
    
    train_out_path.parent.mkdir(parents=True, exist_ok=True)
    eval_out_path.parent.mkdir(parents=True, exist_ok=True)
    
    train_ready.to_parquet(train_out_path, index=False)
    eval_ready.to_parquet(eval_out_path, index=False)
    print(f"[SUCCESS] Data processed and saved.")

if __name__ == "__main__":
    main()
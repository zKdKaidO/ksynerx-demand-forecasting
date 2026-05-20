from pathlib import Path
import pandas as pd
import numpy as np
import os
import yaml

def load_config():
    BASE_DIR = Path(__file__).resolve().parent.parent
    config_path = BASE_DIR / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"[CRITICAL] Can't find config at: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các đặc trưng mới và xử lý ngoại lệ theo chuẩn."""
    processed = df.copy()
    
    # 1. Đảm bảo cột thời gian chuẩn xác
    dates = pd.to_datetime(processed['dt'])
    processed['ds'] = dates
    
    # --- FIX ISSUE 2.2.1: Thêm features week_of_year và year ---
    processed['week_of_year'] = dates.dt.isocalendar().week.astype(int)
    processed['year'] = dates.dt.year
    
    # 2. Xây dựng Khóa chính (Unique ID)
    processed['unique_id'] = processed['store_id'].astype(str) + "_" + processed['product_id'].astype(str)
    
    # --- FIX ISSUE 2.2.2: Tối ưu công thức LDR (Lost Demand Rate) ---
    # Tính số giờ thực tế có hàng trên kệ
    available_hours = 16 - processed['stock_hour6_22_cnt']
    
    # Dùng np.where gộp để xử lý an toàn: Nếu không bán được gì (sale=0), nhu cầu = 0.
    # Ngược lại, upscale nhu cầu dựa trên số giờ có hàng (tránh chia cho 0).
    processed['adjusted_demand'] = np.where(
        processed['sale_amount'] == 0, 
        0.0, 
        processed['sale_amount'] * (16 / np.maximum(available_hours, 1))
    )
    
    # --- FIX ISSUE 2.2.4: Giữ lại giá gốc (nếu dataset có) hoặc đổi tên cho rõ ràng
    # Tùy thuộc vào dataset thực tế của bạn, ở đây giả định discount là selling_price
    processed['selling_price'] = processed['discount']
    
    # Đổi tên cột mục tiêu (y) cho mlforecast
    processed['y'] = processed['adjusted_demand']
    
    return processed

def validate_data(df: pd.DataFrame):
    """FIX ISSUE 2.2.3: Data Quality Validation (Chống dữ liệu rác)"""
    print("[INFO] Running Data Validation...")
    if (df['sale_amount'] < 0).any():
        raise ValueError("Critical Data Error: Found negative sales amounts.")
    if df.duplicated(subset=['unique_id', 'ds']).any():
        print("[WARNING] Found duplicate (store, product, date) records. Dropping duplicates.")
        df = df.drop_duplicates(subset=['unique_id', 'ds'])
    print("[SUCCESS] Data validation passed.")
    return df

def main():
    cfg = load_config()
    
    print("[INFO] Loading raw Train and Eval data...")
    train_df = pd.read_parquet(cfg['data']['files']['train']['raw'])
    eval_df = pd.read_parquet(cfg['data']['files']['eval']['raw'])
    
    print("[INFO] Processing Train data...")
    train_ready = build_features(train_df)
    train_ready = validate_data(train_ready)
    
    print("[INFO] Processing Eval data...")
    eval_ready = build_features(eval_df)
    eval_ready = validate_data(eval_ready)
    
    # Loại bỏ các cột rò rỉ dữ liệu (Data Leakage)
    leakage_cols = ['units_ordered', 'stockout_flag', 'dt', 'adjusted_demand', 'sale_amount']
    drop_cols = [c for c in leakage_cols if c in train_ready.columns]
    
    train_ready = train_ready.drop(columns=drop_cols)
    eval_ready = eval_ready.drop(columns=drop_cols)
    
    train_ready.to_parquet(cfg['data']['files']['train']['processed'], index=False)
    eval_ready.to_parquet(cfg['data']['files']['eval']['processed'], index=False)
    print(f"[SUCCESS] Data processed and saved.")

if __name__ == "__main__":
    main()
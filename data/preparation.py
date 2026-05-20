import os
import pandas as pd
import numpy as np
import yaml

def load_config(config_file: str = "config/config.yaml") -> dict:
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def load_data(file_path: str) -> pd.DataFrame:
    print(f"[INFO] Loading raw data from: {file_path}")
    return pd.read_parquet(file_path)

def filter_city(df: pd.DataFrame, city_id: int) -> pd.DataFrame:
    return df[df['city_id'] == city_id].copy()

def apply_ldr(df: pd.DataFrame) -> pd.DataFrame:
    available_hours = 16 - df['stock_hour6_22_cnt']
    safe_hours = np.where(available_hours == 0, 1, available_hours)
    
    df['adjusted_demand'] = df['sale_amount'] * (16 / safe_hours)
    df.loc[df['sale_amount'] == 0, 'adjusted_demand'] = 0.0
    df['adjusted_demand'] = df['adjusted_demand'].round(2)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    processed = pd.DataFrame()
    
    # Identifiers
    processed['unique_id'] = df['store_id'].astype(str) + "_" + df['product_id'].astype(str)
    processed['store_id'] = df['store_id']
    processed['product_id'] = df['product_id']
    processed['location'] = df['city_id']
    
    # Hierarchy
    processed['management_group'] = df['management_group_id']
    processed['category_1'] = df['first_category_id']
    processed['category_2'] = df['second_category_id']
    processed['category_3'] = df['third_category_id']
    
    # Temporal
    dates = pd.to_datetime(df['dt'])
    processed['ds'] = dates  
    processed['day_of_week'] = dates.dt.dayofweek
    processed['month'] = dates.dt.month
    processed['is_weekend'] = processed['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    processed['is_month_start'] = dates.dt.is_month_start.astype(int)
    processed['is_month_end'] = dates.dt.is_month_end.astype(int)
    processed['holiday'] = df['holiday_flag']
    
    # Context & Weather
    processed['is_promotion'] = df['activity_flag']
    processed['selling_price'] = df['discount']
    processed['precipitation'] = df['precpt']
    processed['temperature'] = df['avg_temperature']
    processed['humidity'] = df['avg_humidity']
    processed['wind_level'] = df['avg_wind_level']
    processed['is_raining'] = df['precpt'].apply(lambda x: 1 if x > 0 else 0)
    processed['is_hot_weather'] = df['avg_temperature'].apply(lambda x: 1 if x > 32 else 0)
    
    # Targets
    processed['units_ordered'] = df['sale_amount']
    processed['stockout_flag'] = df['stock_hour6_22_cnt'].apply(lambda x: 1 if x > 0 else 0)
    processed['adjusted_demand'] = df['adjusted_demand']
    
    return processed

def process_file(raw_path: str, processed_path: str, city_id: int):
    """Execute the full data pipeline for a single file."""
    print(f"\n--- PROCESSING: {raw_path} ---")
    raw_df = load_data(raw_path)
    city_df = filter_city(raw_df, city_id)
    ldr_df = apply_ldr(city_df)
    final_df = build_features(ldr_df)
    
    final_df = final_df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    final_df.to_parquet(processed_path, index=False)
    print(f"[SUCCESS] Saved to: {processed_path}")

def main():
    cfg = load_config()
    target_city = cfg['data']['target_city']
    
    # Process Train Data
    process_file(
        cfg['data']['files']['train']['raw'], 
        cfg['data']['files']['train']['processed'], 
        target_city
    )
    
    # Process Eval Data
    process_file(
        cfg['data']['files']['eval']['raw'], 
        cfg['data']['files']['eval']['processed'], 
        target_city
    )
    
    print("\n[SUCCESS] STAGE 1 COMPLETE FOR BOTH TRAIN AND EVAL!")

if __name__ == "__main__":
    main()
# Fresh Retail Demand Forecasting

An end-to-end Machine Learning pipeline and RESTful API designed to forecast next-day adjusted inventory demand for fresh retail products. This project handles data preparation, time-series model training, and production-ready API serving.

## 1. Dataset Overview

The dataset used for this project is **FreshRetailNet-50K**, publicly available on [Hugging Face](https://huggingface.co/datasets/kSynerX/FreshRetailNet-50K). It consists of highly granular daily transaction logs, segmented into `train.parquet` (approx. 4.5M rows) and `eval.parquet` (approx. 350K rows).

### Key Features & Meaning
* **Identifiers:** `store_id`, `product_id`, `city_id` (location), category hierarchies (`management_group_id`, `first_category_id`, etc.).
* **Temporal:** `dt` (Date of the record).
* **Sales & Inventory:** * `sale_amount`: Actual units sold.
  * `stock_hour6_22_cnt`: Hours during the operating window (06:00 - 22:00) where the product was out of stock.
* **Business Context:** `discount` (price markdown), `activity_flag` (promotions), `holiday_flag`.
* **Weather Data:** `precpt` (precipitation), `avg_temperature`, `avg_humidity`, `avg_wind_level`.

## 2. Data Preprocessing & Feature Engineering

To ensure the model learns true market demand rather than constrained sales, several preprocessing steps were strictly applied:

* **Lost Demand Rate (LDR) Calculation:** The target variable is not raw sales, but `adjusted_demand`. If a product was out of stock for $H$ hours during the 16-hour operating window, the demand was upscaled using the formula: 
  `adjusted_demand = sale_amount * (16 / (16 - H))`
* **Feature Mapping & Extraction:**
  * **Temporal Features:** Extracted `day_of_week`, `month`, `is_weekend`, `is_month_start`, and `is_month_end`.
  * **Weather Binarization:** Mapped continuous weather data into practical flags (`is_raining` if precipitation > 0, `is_hot_weather` if temp > 32°C).
* **Data Leakage Prevention:** Columns that contain future information or raw constraints (`units_ordered`, `stockout_flag`) were aggressively **dropped** from the training pipeline.
* **Type Casting:** Static metadata (like categories and IDs) and temporal flags were explicitly cast to `category` types to optimize tree-based learning.

## 3. Modeling Approach

### Algorithm: LightGBM (via `mlforecast`)
The forecasting engine uses **LightGBMRegressor** wrapped inside Nixtla's `mlforecast` framework. 

**Why LightGBM?**
1. **Speed & Efficiency:** It easily handles millions of rows and computes gradients exceptionally fast, even on CPUs.
2. **Categorical Handling:** It natively supports categorical features without requiring explosive One-Hot Encoding, which is crucial for retail data with thousands of product IDs.
3. **Time-Series Suitability:** Combined with `mlforecast`, it efficiently captures non-linear relationships using historical lags (Lags: 1, 2, 3, 7) and rolling window aggregations (7-day rolling mean).

### Model Evaluation Results (City-Level Scope)
After strictly filtering for the target city and applying outlier capping, the multi-model benchmarking on the held-out dataset yielded:

* **LightGBM (Selected):** MAE = 0.5658 | RMSE = 1.0014 | wMAPE = 42.28%
* **Ridge Baseline:** MAE = 0.5765 | RMSE = 0.9638 | wMAPE = 43.07%
* **Random Forest:** MAE = 0.5858 | RMSE = 1.0556 | wMAPE = 43.77%

**Key Insights:**
* LightGBM generalizes best overall, dominating MAE and volume-weighted error (wMAPE).
* Ridge Regression provided the lowest RMSE, indicating high resistance to extreme outlier predictions.
* Horizon analysis (Walk-forward) confirms stable error rates across the first 3 days (MAE ~0.50 - 0.52), proving the effectiveness of the engineered lag features.

## 4. API Serving & Architecture

The trained model is deployed using **FastAPI** to provide a fast, production-ready, and stateless REST API.

### 💡 API Design Note: Time Series Statefulness & Scenario Simulation
Unlike standard machine learning models, Time-Series forecasting relies heavily on sequential Lags (historical states). It cannot mathematically "jump" to random future dates without generating the intermediate days first. 

To provide a seamless UX while respecting the algorithm's mathematical constraints, the `/predict_next_day` endpoint utilizes a **Scenario Simulation** approach:
1. The API dynamically identifies the chronological "Next Day" based on the model's internal memory using `make_future_dataframe(h=1)`.
2. It overrides the dynamic features (weather, pricing, promotions) of that generated day with the user's JSON payload.
3. It filters out static features to prevent dimensionality conflicts and runs the prediction.

This allows stakeholders to perform **What-If Analysis** (e.g., *"What happens to our demand tomorrow if it rains and we drop the price by 10%?"*) without breaking the time-series continuum.

### 🔒 API Authentication
This API is secured with API Key authentication to simulate production environments.
To test the endpoints via Swagger UI (http://localhost:8000/docs) or cURL, please use the following testing key:

Header Name: X-API-Key

Value: ksynerx-secret-key-2026

## 5. Getting Started

Follow these steps to run the pipeline locally on a Windows/Linux machine.

### Prerequisites
* Python 3.9+
* Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd demand-forecasting
2. **Install dependencies:**
   pip install -r requirements.txt
3. **Run process:**
   python data/preparation.py
   python data/train_model.py


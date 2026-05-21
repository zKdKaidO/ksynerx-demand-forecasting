# Fresh Retail Demand Forecasting

An end-to-end Machine Learning pipeline and RESTful API designed to forecast next-day adjusted inventory demand for fresh retail products. This project handles data preparation, time-series model training, and production-ready API serving.

# Local Structure
```text
demand-forecasting/
├── api/
│   └── serve_api.py             
├── config/
│   └── config.yaml              
├── data/
│   ├── __init__.py
│   ├── eval_ready.parquet       
│   ├── forecasting_data_ready.parquet 
│   ├── preparation.py               # Data Pipeline: Feature engineering, Outlier capping, Validation
│   ├── train_model.py               # Training Pipeline: Multi-model training (LGBM, Ridge, RF)
│   └── train_ready.parquet      
├── models/
│   └── trained_lgbm_pipeline.pkl    
├── venv/                        
├── .gitignore                 
├── README.md                       
└── requirements.txt   
```
To clearly explain how the working prototype functions from end-to-end, here is the lifecycle of the data and model within the system:

1. **The Data Refinery (`preparation.py`):** The prototype starts by ingesting raw parquet files. It does not simply pass data through; it actively corrects business logic (Lost Demand Rate), caps extreme statistical outliers (99.9th percentile), and verifies time-series contiguity (date gap detection).
2. **The Training Engine (`train_model.py`):** Clean data is fed into a multi-model MLForecast pipeline. The prototype trains three different algorithms to benchmark performance. It applies rolling means and standard deviations to capture recent volatility. Crucially, it outputs a highly detailed SRE-standard evaluation (Walk-Forward MAE and wMAPE) to the console, proving its reliability before serializing the best model (LightGBM) to disk.
3. **The Serving Layer (`serve_api.py`):** The prototype spins up a FastAPI server. Upon startup, it preemptively calculates the chronological "Next Day" state for all products and caches it in memory. When a consumer requests a prediction, the API extracts the specific $O(1)$ state for that product, merges it with the dynamic JSON payload (weather, promotions), and returns the forecasted demand.
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

Working with real-world retail data requires moving beyond standard academic datasets. During this phase, I implemented several defensive programming and feature engineering techniques to ensure the data was robust enough for production-grade forecasting.

### 2.1. Reconstructing True Demand (Lost Demand Rate)
In retail, historical sales data only tells us what was actually bought, not what *could* have been bought if the shelves were fully stocked. 
* **The Concept:** If a product sold 10 units but was out of stock for 8 hours out of a 16-hour business day, the recorded sales only represent half of the true daily demand.
* **The Implementation:** I engineered an `adjusted_demand` target using a scaling factor: `sale_amount * (16 / available_hours)`. To ensure mathematical safety and avoid application crashes, a vectorized `np.maximum(available_hours, 1)` was used to strictly prevent "Divide by Zero" errors.

### 2.2. Guarding Against Extreme Outliers (Clipping)
* **The Concept:** The scaling formula above can occasionally produce dangerous outliers. For example, selling 5 items in just 1 available hour inflates the daily demand to 80 items. Tree-based models (like LightGBM) can be overly penalized by these extreme, artificial variances in the target variable.
* **The Implementation:** I applied **Percentile Capping (Clipping)**. Any adjusted demand exceeding the 99.9th percentile is aggressively capped. This bounds the target variable within a realistic physical limit, stabilizing the training loss without deleting valid rows.

### 2.3. Time-Series Contiguity and Lag Integrity
* **The Concept:** Autoregressive models rely heavily on "Lags" (e.g., Lag 1 = yesterday's sales). If a store experiences a system outage and misses a day of data, Friday's record becomes directly adjacent to Sunday's. This silently corrupts the 1-day lag feature, tricking the model.
* **The Implementation:** I built a gap detection mechanism utilizing Pandas `groupby` and `diff().dt.days`. Identifying time gaps `> 1 day` before passing data to the model is a crucial observability practice to ensure the integrity of rolling windows and lags.

### 2.4. Preventing Data Leakage
* **The Concept:** Data leakage is a fatal flaw where a model has access to information during training that it will absolutely not have at prediction time (the future).
* **The Implementation:** After engineering the final target (`y`) and features, all intermediate columns (like `sale_amount`, `discount`, `stock_hour6_22_cnt`) are rigorously dropped. If left in the training set, the model would over-rely on them and completely fail in the production API.

## 3. Modeling Approach & Evaluation Strategy

Moving beyond a basic implementation, I architected a forecasting pipeline that adheres to production standards, focusing on robust evaluation and capturing the true nature of volatile retail data.

### 3.1. Multi-Model Benchmarking
Instead of blindly trusting a single algorithm, I implemented a multi-model pipeline using `mlforecast` to compare different mathematical paradigms simultaneously:
* **LightGBM (Tree-based Ensemble):** Selected for its exceptional speed, native handling of categorical IDs (without the memory explosion of One-Hot Encoding), and ability to capture non-linear trends.
* **Ridge Regression (Linear Baseline):** Included as a conservative, autoregressive baseline. A key lesson learned was that while Tree-based models win on overall accuracy, linear models like Ridge are highly resistant to extreme outliers and rarely make wildly incorrect predictions.
* **Random Forest:** Included to evaluate if bagging (variance reduction) would outperform boosting in this specific highly volatile dataset.

### 3.2. Capturing Volatility via Lag Transforms
Time-series models need more than just raw historical values (lags) to understand context.
* **The Concept:** A product selling 50 units yesterday (Lag 1) could be on a steady trend, or it could be a chaotic spike. The model needs to know the difference.
* **The Implementation:** I engineered rolling window transformations. A `rolling_mean` (7-day and 14-day) teaches the model the recent trend, while a `rolling_std` (Standard Deviation) acts as a "volatility sensor," helping the model distinguish between stable staple goods and erratic fresh produce.

### 3.3. SRE-Standard Evaluation (Walk-Forward & wMAPE)
Academic metrics like a single Global MAE are often deceptive in production. To prove the model's real-world viability, I implemented two critical evaluation techniques:
1. **Walk-Forward Validation (Horizon Breakdown):** A model might predict tomorrow perfectly but fail miserably 3 days from now. By breaking down the MAE by horizon step (Day 1 vs. Day 2 vs. Day 3), I proved that the model maintains its predictive power over time without severe degradation.
2. **Volume-Weighted Metric (wMAPE):** Global MAE treats a 10-unit error on a slow-moving item the same as a 10-unit error on a top-seller. I implemented **wMAPE (Weighted Mean Absolute Percentage Error)** to penalize errors based on actual sales volume, ensuring the evaluation reflects true business impact.

### Model Evaluation Results (City-Level Scope)
After strictly filtering for the target city and applying outlier capping, the multi-model benchmarking on the held-out dataset yielded:

* **LightGBM (Selected):** MAE = 0.5658 | RMSE = 1.0014 | wMAPE = 42.28%
* **Ridge Baseline:** MAE = 0.5765 | RMSE = 0.9638 | wMAPE = 43.07%
* **Random Forest:** MAE = 0.5858 | RMSE = 1.0556 | wMAPE = 43.77%

**Key Insights:**
* LightGBM generalizes best overall, dominating MAE and volume-weighted error (wMAPE).
* Ridge Regression provided the lowest RMSE, indicating high resistance to extreme outlier predictions.
* Horizon analysis (Walk-forward) confirms stable error rates across the first 3 days (MAE ~0.50 - 0.52), proving the effectiveness of the engineered lag features.

## 4. API Serving & Production Architecture

The trained model is deployed as a REST API using **FastAPI**. 

Building a robust, production-ready API exposed me to challenges beyond local scripting. I utilized AI assistance to understand and implement crucial Site Reliability Engineering (SRE) standards:

### Key Architectural Lessons Learned (AI-Assisted)
1. **$O(1)$ Inference via Caching:** A naive implementation would force the model to re-generate the future dataframe for *all* 30,000+ products on every single request, causing severe CPU bottlenecks. The solution was to pre-compute the base dataframe *once* at startup and filter it strictly to the requested $O(1)$ row before copying.
2. **Concurrency & Event Loop Safety:** Machine Learning inference is heavily CPU-bound. If executed directly in an `async def` endpoint, it would block FastAPI's event loop, crashing the server under load. The fix was to offload the prediction task to a dedicated thread pool using `run_in_executor`.
3. **Application Lifespan Management:** To prevent memory leaks and avoid the anti-pattern of global variables, the model and cache are safely initialized and destroyed using FastAPI's `@asynccontextmanager` lifespan events.

### 🔒 API Authentication
To simulate a consumer-facing production environment, the endpoints are secured against unauthorized access. If you are testing the API via Swagger UI (`http://localhost:8000/docs`) or cURL, you **MUST** authorize using this testing key:

* **Header Name:** `X-API-Key`
* **Value:** `ksynerx-secret-key-2026`

## 5. Getting Started

Follow these steps to run the end-to-end pipeline locally. The project uses absolute pathing resolution (`pathlib`), so you can safely run these commands from the project root directory.

### Prerequisites
* Python 3.9+
* Git
* *Recommended:* A Python virtual environment (`venv` or `conda`)

### Installation

1. **Clone the repository & setup environment:**
   ```bash
   git clone <your-repository-url>
   cd demand-forecasting
   
   # Optional but recommended: Create and activate a virtual environment
   python -m venv venv
   # On Windows: venv\Scripts\activate
   # On Linux/Mac: source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

**Step 1: Data Preparation**
Run the data pipeline to clean the raw data, extract features, and apply outlier clipping.
```bash
python data/preparation.py
```
*🎯 **Output:** Generates `train_ready.parquet` and `eval_ready.parquet` inside the `data/` directory.*

**Step 2: Model Training**
Execute the multi-model training pipeline. This will train LightGBM, Ridge, and Random Forest, evaluate them via Walk-Forward validation, and save the best pipeline.
```bash
python data/train_model.py
```
*🎯 **Output:** Generates the serialized model file `trained_lgbm_pipeline.pkl` inside the `models/` directory.*

**Step 3: Start the API Server**
Launch the FastAPI application using Uvicorn.
```bash
uvicorn api.serve_api:app --reload
```
*🎯 **Output:** Starts a local web server running on `http://127.0.0.1:8000`.*

### Testing the API (Swagger UI)

FastAPI automatically generates an interactive documentation page where you can test the predictions directly from your browser.

1. Open your web browser and navigate to: **http://localhost:8000/docs**
2. **Authenticate:** * Click the green **Authorize** button (🔒) on the top right of the page.
   * In the `Value` field, enter the testing API Key: `ksynerx-secret-key-2026`
   * Click **Authorize** and then **Close**.
3. **Run a Health Check:**
   * Expand the `GET /health` endpoint and click **Try it out** -> **Execute** to verify the model is loaded.
4. **Make a Prediction:**
   * Expand the `POST /predict_next_day` endpoint.
   * Click **Try it out**.
   * A sample JSON payload is already pre-filled. Simply click the blue **Execute** button.
   * Scroll down to the **Server response** section to see the forecasted demand!

## 6. Feature Implementation Status

To provide full transparency on the development process, below is a breakdown of which features were successfully delivered and which were deferred.

### ✅ Considered and Implemented
* **Robust Data Quality Gates:** Automated detection of date gaps, negative sales anomalies, and strict isolation of data leakage columns.
* **Stateless, Zero-Global-Variable API:** The serving layer was strictly designed without global variables, utilizing FastAPI's `lifespan` context managers to handle model loading and cache state safely.
* **Multi-Model Benchmarking:** Implementation of LightGBM (Primary), Ridge Regression, and Random Forest for comprehensive algorithmic comparison.
* **Advanced Evaluation Metrics:** Shifting from standard MAE to volume-weighted MAE (wMAPE) and Walk-Forward horizon breakdowns.
* **$O(1)$ Memory-Safe Inference:** Filtering the pre-computed future dataframe *before* copying to prevent memory bloat under concurrent API requests.

### 🚧 Considered but Not Finished (Future Work)
* **Distributed Computing with Ray:** The original specification suggested using `Ray Serve` and `Ray Train` for scalable data management and serving. Due to time constraints and the complexity of configuring a local Ray cluster, I opted for a highly optimized single-node FastAPI architecture. Transitioning this FastAPI logic to Ray Serve would be the immediate next step for horizontal scaling.
* **ARIMA / ETS Statistical Baselines:** While linear (Ridge) and tree-based models were implemented, classic statistical models (like `statsforecast` AutoARIMA) were considered but left out to prioritize pipeline stability and training speed on local hardware.
* **Automated Hyperparameter Tuning:** Optuna was considered for dynamic hyperparameter optimization. However, to ensure the prototype runs quickly for evaluators, I manually constrained the LightGBM parameters (`num_leaves`, `min_child_samples`) based on domain knowledge instead of implementing an exhaustive search.


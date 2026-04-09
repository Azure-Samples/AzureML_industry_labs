# ⚡ Conformal Energy Forecasting

Forecast hourly electricity demand with statistically guaranteed prediction intervals using Conformalized Quantile Regression (CQR), deployed as a batch endpoint on Azure Machine Learning.

---

## Problem Statement

A regional energy utility must forecast **hourly electricity demand 24 hours ahead** to plan generation capacity and spinning reserves. Point forecasts alone are insufficient — grid operators need **calibrated prediction intervals** with formal coverage guarantees. Overestimating demand wastes fuel and increases costs; underestimating it risks brownouts and expensive emergency peaker plant activation.

Traditional approaches rely on distributional assumptions (e.g., Gaussian residuals) that rarely hold for energy demand data, which exhibits heteroscedastic noise, multiple seasonalities, and heavy tails during extreme weather events.

**Conformal prediction** — specifically **Conformalized Quantile Regression (CQR)** — provides distribution-free prediction intervals with rigorous finite-sample coverage guarantees: if you request 90% coverage, at least 90% of true values will fall within the interval, regardless of the underlying data distribution. This makes it ideal for safety-critical operational decisions in energy systems.

---

## Dataset

This lab uses a **synthetic dataset** that models realistic electricity demand patterns for a mid-sized regional utility:

| Property | Value |
|----------|-------|
| **Duration** | 3 years (26,280 hourly observations) |
| **Target** | Electricity demand in MWh |
| **Range** | ~50 – 1,200+ MWh |
| **Format** | Generated in-pipeline, saved as `.pt` tensors |

### Demand Patterns Modelled

- **Daily seasonality** — morning peak (7–9am), evening peak (5–8pm), overnight trough
- **Weekly seasonality** — ~15% lower demand on weekends
- **Annual seasonality** — higher in summer (AC) and winter (heating)
- **Temperature correlation** — U-shaped (extreme temps → HVAC demand)
- **Upward trend** — simulating population growth and electrification (~2%/year)
- **Heteroscedastic noise** — higher variance during peak hours (6am–10pm)

### Engineered Features (10)

| Feature | Description |
|---------|-------------|
| `hour` | Hour of day (0–23) |
| `day_of_week` | Day of week (0–6) |
| `month` | Month (1–12) |
| `is_weekend` | Binary weekend flag |
| `temperature` | Ambient temperature (°C) |
| `demand_lag_1` | Demand at t-1 |
| `demand_lag_24` | Demand at t-24 (same hour yesterday) |
| `demand_lag_168` | Demand at t-168 (same hour last week) |
| `demand_roll_mean_24` | 24-hour rolling mean |
| `demand_roll_mean_168` | 168-hour (weekly) rolling mean |

### Splits (Chronological)

| Split | Proportion | Purpose |
|-------|-----------|---------|
| Train | 60% | Model training |
| Calibration | 20% | Conformal score computation |
| Test | 20% | Final evaluation |

---

## Model Architecture

**QuantileForecaster** — a multi-output MLP with three parallel quantile heads:

```
Input (10 features)
  → Linear(128) → ReLU → Dropout(0.2)
  → Linear(64)  → ReLU → Dropout(0.2)
  → Three parallel heads:
      head_lower  → Linear(1)   # predicts α/2 quantile (0.05)
      head_median → Linear(1)   # predicts 0.50 quantile
      head_upper  → Linear(1)   # predicts 1-α/2 quantile (0.95)
```

All three heads share the same backbone and are trained jointly with **pinball (quantile) loss**.

### Conformal Prediction Method: CQR

After training, the model's quantile predictions are **conformalized** on a held-out calibration set:

1. Compute nonconformity scores on calibration data:
   $$E_i = \max\bigl(\hat{q}_{lo}(X_i) - Y_i,\; Y_i - \hat{q}_{hi}(X_i)\bigr)$$

2. Compute the conformal threshold:
   $$\hat{Q} = \text{Quantile}_{(1-\alpha)(1 + 1/n_{\text{cal}})}\bigl(\{E_1, \dots, E_{n_{\text{cal}}}\}\bigr)$$

3. At inference, produce guaranteed prediction intervals:
   $$C(X) = \bigl[\hat{q}_{lo}(X) - \hat{Q},\; \hat{q}_{hi}(X) + \hat{Q}\bigr]$$

This provides marginal coverage ≥ 1−α without any distributional assumptions.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Learning Rate** | 1e-3 |
| **Batch Size** | 256 |
| **Max Epochs** | 100 |
| **Early Stopping** | Patience 15 (on val loss) |
| **Optimizer** | Adam |
| **Scheduler** | ReduceLROnPlateau (patience=5, factor=0.5) |
| **Loss** | Pinball (quantile) loss |
| **Validation Split** | 15% of training set |
| **Coverage Target (α)** | 0.10 (90% coverage) |
| **Lower Quantile** | 0.05 |
| **Upper Quantile** | 0.95 |

---

## Prerequisites

- Azure subscription with Contributor access
- Azure ML workspace with a **GPU** compute cluster (named `gpu1`)
- Azure CLI with ML extension (`az extension add -n ml`)
  - Ensure the v1 extension is **not** installed: `az extension remove -n azure-cli-ml`
- Python 3.10+ with a virtual environment
- System-assigned managed identity (SAMI) on the compute cluster with **Contributor** role on the workspace
- Dependencies from `requirements.txt`

---

## How to Run

### 1. Build and register the custom environment (one-time)

```bash
cd conformal_energy_forecasting
az ml environment create \
  --name energy-forecast-env \
  --build-context . \
  --dockerfile-path Dockerfile \
  --resource-group <rg> \
  --workspace-name <ws>
```

### 2. Submit the pipeline

```bash
python main.py
```

### 3. Monitor in Azure ML Studio

The Studio URL is printed on submission. Track:
- Per-epoch training/validation loss curves
- Conformal calibration metrics
- Test coverage vs. target coverage
- Mean prediction interval width

---

## Running Inference

Once the pipeline completes and the batch endpoint is deployed, you can invoke it to generate forecasts with conformal prediction intervals.

### Prerequisites

- The pipeline must have run successfully with a deployed batch endpoint (`energy-forecast-batch`)
- Input data must be `.pt` files, each containing a `features` tensor of shape `(10,)` with normalised features
- Features must be normalised using the same `norm_stats.json` produced during preprocessing

### Input Format

Each `.pt` file should contain a dictionary with:

```python
{
    "features":  torch.tensor([...], dtype=torch.float32),  # shape (10,) — normalised
    "timestamp": "2024-03-15_14",                           # optional, for output labelling
    "target":    torch.tensor(0.5, dtype=torch.float32),    # optional, for coverage evaluation
}
```

The 10 features (in order): `hour`, `day_of_week`, `month`, `is_weekend`, `temperature`, `demand_lag_1`, `demand_lag_24`, `demand_lag_168`, `demand_roll_mean_24`, `demand_roll_mean_168` — all normalised (zero-mean, unit-variance) using the statistics from training.

### Preparing Input Data

A sample payload generator is included to help you test the endpoint quickly. It creates 24 `.pt` files representing a next-day forecast for a typical summer weekday:

```bash
# Using exact normalisation stats from your training run
python generate_sample_payload.py --norm-stats <path-to-norm_stats.json>

# Or using approximate defaults (good enough for a quick test)
python generate_sample_payload.py
```

This produces a `sample_payload/` folder with 24 files (one per hour). You can also pass `--start-date 2024-12-01` to simulate a winter day or `--output-dir my_folder` to change the output location.

Upload the payload as a data asset:

```bash
az ml data create \
  --name energy-inference-input \
  --type uri_folder \
  --path ./sample_payload/ \
  --resource-group <rg> \
  --workspace-name <ws>
```

> **Tip:** To use your own real data, create `.pt` files with the same structure. Each file must contain a `features` tensor of shape `(10,)` normalised using `norm_stats.json` from the training run. Download this file from the registered model artefacts in Azure ML Studio.

### Invoking the Batch Endpoint

#### Via Azure CLI

```bash
az ml batch-endpoint invoke \
  --name energy-forecast-batch \
  --deployment-name energy-forecast-deployment \
  --input azureml:energy-inference-input:1 \
  --resource-group <rg> \
  --workspace-name <ws>
```

> Replace `:1` with the version number returned by `az ml data create`. The `--deployment-name` flag is required unless you have set a default deployment.

#### Via Python SDK

```python
from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

job = ml_client.batch_endpoints.invoke(
    endpoint_name="energy-forecast-batch",
    input=Input(type=AssetTypes.URI_FOLDER, path="azureml:energy-inference-input:1"),
)
print(f"Batch job submitted: {job.name}")

# Wait for completion
ml_client.jobs.stream(job.name)
```

### Output Format

The endpoint produces a `predictions.csv` with a header row followed by one row per input `.pt` file. Non-`.pt` files in the input are skipped.

| Column | Description |
|--------|-------------|
| `timestamp` | Timestamp label from the input file (e.g., `2024-07-15_14`) |
| `point_forecast` | Median demand forecast (MWh) — your best single estimate |
| `raw_lower` | Model’s raw 5th percentile prediction (MWh) — before conformal adjustment |
| `raw_upper` | Model’s raw 95th percentile prediction (MWh) — before conformal adjustment |
| `conformal_lower` | Calibrated lower bound (MWh) — use this for operational planning |
| `conformal_upper` | Calibrated upper bound (MWh) — use this for operational planning |
| `interval_width` | `conformal_upper - conformal_lower` (MWh) — wider = more uncertainty |
| `actual` | True demand if provided in input `.pt` file, otherwise empty |

#### Sample Output

```
timestamp       point_forecast  raw_lower  raw_upper  conformal_lower  conformal_upper  interval_width  actual
2024-07-15_00   745.96          689.30     814.86     706.18           797.98           91.80
2024-07-15_01   697.44          642.94     763.57     659.82           746.69           86.88
2024-07-15_02   680.12          625.88     745.21     642.76           728.33           85.57
```

> **Note:** The `actual` column is empty when input files don’t include a `target` tensor. It is populated when using test split data from the training pipeline.

### Downloading Results

```bash
# Get the output URL from the completed job
az ml job show --name <job-name> --query outputs --resource-group <rg> --workspace-name <ws>

# Download predictions
az ml job download --name <job-name> --output-name score --download-path ./results/ \
  --resource-group <rg> --workspace-name <ws>
```

### Interpreting the Results

- **`point_forecast`** — use this as the central demand estimate for planning
- **`conformal_lower` / `conformal_upper`** — the 90% prediction interval with formal coverage guarantees. At least 90% of true demand values will fall within `[conformal_lower, conformal_upper]`
- **`raw_lower` / `raw_upper`** — the model’s uncalibrated quantile predictions (for comparison — these lack coverage guarantees)
- **`interval_width`** — wider intervals indicate higher uncertainty (e.g., extreme weather, unusual hours)

> **Important:** Input features must be normalised using the exact `norm_stats.json` from the training run. Using approximate or mismatched normalisation statistics will produce inaccurate forecasts. Download `norm_stats.json` from the registered model artefacts in Azure ML Studio.

---

## Outputs

| Artefact | Type | Description |
|----------|------|-------------|
| Processed tensors | Azure ML Data Asset (`URI_FOLDER`) | Train, calibration, and test `.pt` tensors with normalisation stats |
| Trained model weights | Azure ML Model | `quantile_forecaster.pth` + `conformal_Q.json` + `norm_stats.json` |
| MLflow experiment | Experiment runs | Per-epoch loss, coverage, interval width, conformal threshold |
| Batch endpoint | `energy-forecast-batch` | Produces `predictions.csv` with conformal prediction intervals |

---

## Project Structure

```
conformal_energy_forecasting/
├── main.py                       # Pipeline orchestration & job submission
├── lab.json                      # Lab metadata (CI auto-updates root README)
├── requirements.txt              # Python dependencies (pinned versions)
├── Dockerfile                    # Custom Azure ML environment image
├── README.md                     # Lab-specific documentation
├── generate_sample_payload.py    # Utility to create test inference inputs
├── .amlignore                    # Files excluded from Azure ML snapshots
├── .gitignore                    # Files excluded from version control
├── data_processing/
│   ├── __init__.py
│   └── preprocess.py             # Synthetic data generation & feature engineering
├── model/
│   ├── __init__.py
│   └── quantile_forecaster.py    # QuantileForecaster model architecture
└── pipeline/
    ├── preprocess_step.py        # Data preprocessing pipeline step
    ├── train_step.py             # Model training + conformal calibration step
    ├── register_model.py         # Model registration with gating step
    ├── deploy_endpoint.py        # Batch endpoint deployment step
    └── score.py                  # Scoring script for batch inference
```

---

## Tech Stack

| Technology | Version / Detail |
|------------|-----------------|
| Python | 3.10+ |
| PyTorch | 2.1.0 |
| MLflow | 2.9.2 |
| Azure ML SDK | v2 (`azure-ai-ml`) |
| Experiment Tracking | `azureml-mlflow` |
| Authentication | `DefaultAzureCredential` (`azure-identity`) |
| Compute | Azure ML managed GPU cluster |
| Deployment | Azure ML Batch Endpoints |
| Base Image | `openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04` |

---
| Predictions | `predictions.csv` | Columns: timestamp, point_forecast, raw_lower, raw_upper, conformal_lower, conformal_upper, interval_width, actual |

---

## Project Structure

```
conformal_energy_forecasting/
├── main.py                      # Pipeline orchestration & job submission
├── lab.json                     # Lab metadata for CI
├── requirements.txt             # Pinned Python dependencies
├── Dockerfile                   # Custom Azure ML environment (GPU)
├── README.md                    # This file
├── generate_sample_payload.py   # Generate sample .pt files for testing inference
├── .amlignore                   # Azure ML snapshot exclusions
├── .gitignore                   # Git exclusions
├── data_processing/
│   ├── __init__.py
│   └── preprocess.py            # EnergyDemandDataset class
├── model/
│   ├── __init__.py
│   └── quantile_forecaster.py   # QuantileForecaster (PyTorch nn.Module)
└── pipeline/
    ├── preprocess_step.py       # Synthetic data generation + feature engineering
    ├── train_step.py            # Quantile regression training + CQR calibration
    ├── register_model.py        # Model comparison + registration
    ├── deploy_endpoint.py       # Batch endpoint deployment
    └── score.py                 # Inference: point forecasts + conformal intervals
```

---

## Tech Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Language |
| PyTorch | 2.1.0 | Model training |
| Azure ML SDK v2 | latest | Pipeline orchestration |
| MLflow | 2.9.2 | Experiment tracking |
| NumPy | <2.0 | Numerical operations, CQR calibration |
| Pandas | latest | Inference output formatting |

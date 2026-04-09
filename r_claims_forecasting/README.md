# 📊 R Claims Severity Forecasting

Predict insurance claims severity using an R Gamma GLM deployed as a batch endpoint on Azure Machine Learning.

---

## Problem Statement

An insurance company needs to predict the **severity** (dollar amount) of motor insurance claims to set accurate reserves and price policies. Claims amounts are positive, right-skewed, and vary based on policyholder demographics, vehicle characteristics, and risk history.

The industry-standard approach uses **Generalized Linear Models (GLMs)** with a **Gamma family** and **log link** — a well-established actuarial technique that:

- Naturally handles positive, continuous, right-skewed response variables
- Provides interpretable coefficients (multiplicative effects on the log scale)
- Produces standard errors and prediction intervals from the model fit
- Is required by many insurance regulators for rate-filing documentation

This lab demonstrates running an **R statistical model** end-to-end on Azure ML: from data generation through training, model registration with gating, and batch endpoint deployment.

---

## Dataset

This lab uses a **synthetic dataset** modelling motor insurance policies:

| Property | Value |
|----------|-------|
| **Policies** | 50,000 |
| **Claims rate** | ~15–45% (risk-dependent) |
| **Severity range** | $100 – $10,000+ |
| **Format** | Generated in-pipeline, saved as CSV |

### Policyholder Features

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Policyholder age (18–80) |
| `gender` | Categorical | M / F |
| `vehicle_age` | Numeric | Vehicle age in years (0–20) |
| `vehicle_value` | Numeric | Vehicle value ($5,000–$80,000) |
| `region` | Categorical | urban / suburban / rural |
| `credit_score` | Numeric | Credit score (300–850) |
| `n_prior_claims` | Numeric | Number of prior claims (Poisson distributed) |
| `coverage_type` | Categorical | basic / standard / premium |
| `policy_tenure` | Numeric | Years with insurer (0–15) |

### Risk Factors Modelled

- Young drivers (<25) and elderly drivers (>70) have higher claim probability
- Urban regions have higher frequency and severity
- Low credit scores correlate with higher claim rates
- Prior claims history increases future claim likelihood
- Vehicle value and coverage type affect severity

### Splits (Chronological)

| Split | Proportion | Purpose |
|-------|-----------|---------|
| Train | 60% | Model fitting |
| Validation | 20% | Model evaluation |
| Test | 20% | Final metrics + model gating |

---

## Model Architecture

**Gamma GLM** — the standard actuarial severity model:

```
claim_amount ~ age + I(age²) + gender + vehicle_age + I(vehicle_age²) +
               log(vehicle_value) + region + credit_score + n_prior_claims +
               coverage_type + policy_tenure

Family: Gamma
Link:   log (ensures positive predictions)
```

### Key Properties

- **Gamma family**: models positive, right-skewed data — natural for claim amounts
- **Log link**: coefficients represent multiplicative effects (e.g., urban region multiplies expected severity by exp(β))
- **Polynomial terms**: `age²` and `vehicle_age²` capture non-linear effects
- **Log transform**: `log(vehicle_value)` captures diminishing marginal effect of vehicle value

### Prediction Intervals

90% prediction intervals are computed using the standard error of the linear predictor:

$$\hat{y}_{lower} = \exp\bigl(\hat{\eta} - z_{0.95} \cdot \text{SE}(\hat{\eta})\bigr), \quad \hat{y}_{upper} = \exp\bigl(\hat{\eta} + z_{0.95} \cdot \text{SE}(\hat{\eta})\bigr)$$

where $\hat{\eta}$ is the linear predictor and $z_{0.95} = 1.645$.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Family** | Gamma |
| **Link function** | log |
| **Fitting method** | IRLS (Iteratively Reweighted Least Squares) |
| **Model selection** | AIC / BIC |
| **Gating metric** | Test MAE (must improve on registered model) |

---

## Prerequisites

- Azure subscription with Contributor access
- Azure ML workspace with a **CPU** compute cluster (named `cpu1`)
- Azure CLI with ML extension (`az extension add -n ml`)
  - Ensure the v1 extension is **not** installed: `az extension remove -n azure-cli-ml`
- Python 3.10+ with a virtual environment
- System-assigned managed identity (SAMI) on the compute cluster with **Contributor** role on the workspace
- Dependencies from `requirements.txt`

> **Note:** R is installed in the Docker environment — you do **not** need R installed locally.

---

## How to Run

### 1. Build and register the custom environment (one-time)

```bash
cd r_claims_forecasting
az ml environment create \
  --name r-claims-forecast-env \
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
- R console output (model summary, coefficient table)
- Train / validation / test metrics (MAE, RMSE, MAPE, deviance)
- AIC and BIC for model complexity assessment
- Dispersion parameter and Gamma shape

---

## Running Inference

### Input Format

Input data should be a CSV file with the same columns as the training data (minus `claim_amount` and `has_claim`, which are predicted):

```csv
policy_start,age,gender,vehicle_age,vehicle_value,region,credit_score,n_prior_claims,coverage_type,policy_tenure
2024-06-15,35,M,3,45000,urban,720,0,standard,5
2024-06-15,22,F,8,18000,suburban,650,1,basic,1
```

### Invoking the Batch Endpoint

```bash
# Upload input data
az ml data create \
  --name claims-inference-input \
  --type uri_folder \
  --path ./inference_data/ \
  --resource-group <rg> \
  --workspace-name <ws>

# Invoke the endpoint
az ml batch-endpoint invoke \
  --name r-claims-severity-batch \
  --deployment-name r-claims-severity-deployment \
  --input azureml:claims-inference-input:1 \
  --resource-group <rg> \
  --workspace-name <ws>
```

### Output Format

| Column | Description |
|--------|-------------|
| `predicted_severity` | Point estimate of claim amount ($) |
| `lower_90` | Lower bound of 90% prediction interval ($) |
| `upper_90` | Upper bound of 90% prediction interval ($) |
| `interval_width` | `upper_90 - lower_90` ($) |
| `policy_start` | Policy start date (from input) |
| `age` | Policyholder age (from input) |
| `actual_claim_amount` | True claim amount if provided (for evaluation) |

---

## Outputs

| Artefact | Type | Description |
|----------|------|-------------|
| Processed CSVs | Azure ML Data Asset (`URI_FOLDER`) | Train, val, test splits with claims-only subsets |
| Trained model | Azure ML Model | `model.rds` + `norm_stats.json` + `formula.txt` + `metrics.json` |
| Batch endpoint | `r-claims-severity-batch` | Produces `predictions.csv` with severity predictions and intervals |

---

## Project Structure

```
r_claims_forecasting/
├── main.py                      # Pipeline orchestration & job submission
├── lab.json                     # Lab metadata for CI
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Custom environment (Ubuntu + R + Python)
├── README.md                    # This file
├── generate_sample_payload.py   # Generate sample CSV for batch inference
├── .amlignore                   # Azure ML snapshot exclusions
├── .gitignore                   # Git exclusions
├── sample_payload/              # Generated sample CSV (git-ignored)
├── data_processing/
│   ├── __init__.py
│   └── preprocess.py            # ClaimsSeverityDataset helper class
├── model/
│   ├── __init__.py
│   └── gamma_glm.R              # Gamma GLM model definition (R)
└── pipeline/
    ├── preprocess_step.R        # Synthetic data generation (R)
    ├── train_step.R             # Model training + evaluation (R)
    ├── register_model.py        # Model comparison + registration
    ├── deploy_endpoint.py       # Batch endpoint deployment
    ├── score.py                 # Python scoring wrapper (init/run)
    └── score_predict.R          # R prediction script (called by score.py)
```

---

## Tech Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| R | 4.x | Statistical modelling (Gamma GLM) |
| Python | 3.10+ | Pipeline orchestration, batch scoring wrapper |
| Azure ML SDK v2 | latest | Pipeline orchestration |
| Azure CLI | latest | Environment and endpoint management |
| argparse (R) | latest | Command-line argument parsing in R scripts |
| jsonlite (R) | latest | JSON I/O for metrics and normalisation stats |
| Pandas | latest | Inference output formatting |

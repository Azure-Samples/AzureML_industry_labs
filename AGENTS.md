# AGENTS.md

This document describes the workspace structure, conventions, and patterns for AI coding agents working in the Azure ML Industry Labs repository. Use this as context when generating, modifying, or reviewing labs.

## Overview

**Azure ML Industry Labs** (`azureml_industry_labs`) is a collection of end-to-end machine learning labs across different industry use-cases, built on **Azure Machine Learning**. Each lab demonstrates a complete MLOps workflow — data preprocessing, model training, model registration, and deployment — using the Azure ML SDK v2, MLflow experiment tracking, and managed endpoints.

Labs are self-contained Python projects (not notebooks) with a shared structural convention.

---

## Directory Structure

### Root

```
azureml_industry_labs/
├── README.md              # Repo overview, labs table, quick start
├── CONTRIBUTING.md         # Guidelines for adding new labs
├── AGENTS.md               # This file — workspace context for AI agents
├── LICENSE                 # MIT License
├── images/                 # Shared images and diagrams
│
└── <lab_name>/             # Each lab is a top-level directory
```

### Lab Structure (Standard Template)

Every lab **must** follow this structure. When creating a new lab, replicate this layout exactly:

```
<lab_name>/
├── main.py                  # Pipeline orchestration & job submission
├── lab.json                 # Lab metadata (used by CI to update root README)
├── requirements.txt         # Python dependencies (pinned versions)
├── Dockerfile               # Custom Azure ML environment image
├── README.md                # Lab-specific documentation
├── .amlignore               # Files to exclude from Azure ML snapshots
├── .gitignore               # Files to exclude from version control
├── data_processing/
│   ├── __init__.py
│   └── preprocess.py        # Reusable dataset class (PyTorch Dataset)
├── model/
│   ├── __init__.py
│   └── <model_name>.py      # Model architecture definition
└── pipeline/
    ├── preprocess_step.py   # Data preprocessing pipeline step
    ├── train_step.py        # Model training pipeline step
    ├── register_model.py    # Model registration pipeline step
    ├── deploy_endpoint.py   # Endpoint deployment pipeline step
    └── score.py             # Scoring script for batch/online inference
```

### Existing Labs

| Directory | Industry | Description |
|-----------|----------|-------------|
| `cpp_model_training/` | Industry Agnostic | Train a linear regression model in pure C++ with Azure ML pipeline orchestration |
| `conformal_energy_forecasting/` | Energy & Utilities | Forecast hourly electricity demand with calibrated prediction intervals using Conformalized Quantile Regression (CQR) deployed via Azure ML batch endpoints |
| `azureml_mcp_server/` *(external)* | Industry Agnostic | Deploy a trained ML model to Azure ML and expose it as an MCP server via APIM for Foundry agents |

### External Labs

Labs hosted in other repositories can be linked into this repo without duplicating code. Create a stub directory containing only:

```
<lab_name>/
├── lab.json       # Lab metadata with a "githubPath" field pointing to the external repo
└── README.md      # One-liner description with a link to the external repo
```

The `githubPath` field in `lab.json` overrides the default link in both the root README table and the GitHub Pages site. The CI script sets `"external": true` in the pages config, which displays an "External" badge on the lab card.

---

## Key Conventions

### Pipeline Orchestration (`main.py`)

- Use the Azure ML SDK v2 `@pipeline` decorator to define the pipeline DAG.
- Define each step as a `command()` component with explicit `inputs`, `outputs`, `environment`, and `compute`.
- Use `Input(type=AssetTypes.URI_FOLDER)` and `Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")` for data passing between steps.
- Submit with `ml_client.jobs.create_or_update()` and stream logs with `ml_client.jobs.stream()`.
- Connect to the workspace via `MLClient.from_config(credential=DefaultAzureCredential())`.

```python
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

@pipeline(name="my_pipeline", description="...")
def my_pipeline(raw_data):
    preprocess_job = preprocess_component(raw_data=raw_data)
    train_job = train_component(processed_data=preprocess_job.outputs.processed_data)
    register_job = register_component(model_output=train_job.outputs.model_output)
    deploy_component(register_output=register_job.outputs.register_output)
```

### Pipeline Steps (`pipeline/`)

Each step is a standalone Python script invoked via `command()`. Steps communicate through mounted folder I/O and flag files.

| Step | File | Responsibility |
|------|------|----------------|
| **Preprocess** | `preprocess_step.py` | Transform raw data into model-ready format. Save as `.pt` tensors (or equivalent). Generate manifests for change detection. |
| **Train** | `train_step.py` | Train the model. Log all metrics and parameters to MLflow. Save the best model weights. Write status flags for downstream steps. |
| **Register** | `register_model.py` | Compare new model against the existing registered version. Only register + gate deployment if the new model is better. Tag the model with full lineage metadata. |
| **Deploy** | `deploy_endpoint.py` | Create or update a batch or online endpoint. Only deploy if the register step signals improvement. |
| **Score** | `score.py` | Implement `init()` and `run(mini_batch)` for batch endpoints, or a Flask/FastAPI app for online endpoints. |

### Incremental Processing

Labs should implement smart pipeline behaviour to avoid redundant work:

- **Manifest tracking:** Maintain a JSON manifest of processed files. Compare against previous versions to detect additions and removals.
- **Change flags:** Write boolean flag files (e.g., `training_changed.flag`, `model_trained.flag`, `deploy.flag`) so downstream steps can skip work when nothing changed.
- **Model gating:** Only deploy a new model if its validation accuracy exceeds the currently registered model.

### Experiment Tracking (MLflow)

- Start an MLflow run inside `train_step.py` using `mlflow.start_run()`.
- Log all hyperparameters with `mlflow.log_params()`.
- Log per-epoch metrics (loss, accuracy) with `mlflow.log_metrics()`.
- Log final evaluation metrics including per-class precision, recall, and F1 scores.
- The `azureml-mlflow` package integrates MLflow with the Azure ML workspace automatically.

### Model Registration

- Register models via `ml_client.models.create_or_update()`.
- Tag models with metadata for lineage and comparison:
  - `val_acc`, `test_acc` — Performance metrics
  - `trained_on_asset`, `trained_on_version` — Training data asset and version
  - `tested_on_asset`, `tested_on_version` — Test data asset and version
  - `training_changed`, `testing_changed`, `model_retrained` — Pipeline state flags

### Custom Environments

- Base images should come from the Azure ML curated image catalogue (e.g., `mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04`).
- The `Dockerfile` should `COPY requirements.txt` and `pip install` dependencies.
- Register the environment with `az ml environment create --name <env-name> --build-context . --dockerfile-path Dockerfile`.
- Reference it in pipeline components as `<env-name>@latest`.

### `.amlignore`

Exclude large or unnecessary files from Azure ML snapshots to keep upload times fast:

```
*.pt
*.pth
*.bin
*.jpg
*.jpeg
*.png
*.csv
*.ipynb
__pycache__/
.git/
Dockerfile
.azureml/
```

---

## Lab README Template

Each lab's `README.md` should include these sections:

1. **Title and one-line description**
2. **Problem Statement** — Industry context and what the lab demonstrates
3. **Dataset** — Format, structure, class labels, how to obtain/upload
4. **Model Architecture** — Architecture summary, custom head, key design choices
5. **Training Configuration** — Hyperparameters table (learning rate, optimizer, scheduler, batch size, epochs, etc.)
6. **Data Augmentation** — Transforms applied (if applicable)
7. **Prerequisites** — Azure resources, CLI tools, Python version, compute cluster
8. **How to Run** — Numbered steps: environment setup, data upload, pipeline submission, monitoring
9. **Outputs** — Table of artefacts produced (data assets, model, endpoint, inference results)
10. **Project Structure** — Annotated file tree
11. **Tech Stack** — Table of technologies and versions

---

## Key Technologies

- **Language:** Python 3.10+
- **Pipeline SDK:** Azure ML SDK v2 (`azure-ai-ml`)
- **Experiment Tracking:** MLflow via `azureml-mlflow`
- **Authentication:** `DefaultAzureCredential` from `azure-identity`
- **Compute:** Azure ML managed compute clusters (CPU/GPU)
- **Deployment:** Azure ML Batch Endpoints and/or Managed Online Endpoints
- **Environments:** Custom Docker images registered in Azure ML
- **Infrastructure:** Azure CLI with the `ml` extension

## Prerequisites

- Python 3.10+ with a virtual environment
- Azure CLI authenticated to an Azure subscription (`az login`)
- Azure ML CLI extension (`az extension add -n ml`)
- Azure subscription with Contributor access
- An Azure ML workspace with at least one compute cluster configured

---

## Creating a New Lab

When asked to create a new lab, follow these steps:

1. **Create the lab directory** at the repo root using `snake_case` naming (e.g., `demand_forecasting/`).
2. **Scaffold the standard structure** — copy the template above and populate every file.
3. **Implement `main.py`** with the 4-step pipeline pattern (preprocess → train → register → deploy).
4. **Implement each pipeline step** as a standalone script in `pipeline/`.
5. **Define the model** in `model/<model_name>.py` as a PyTorch `nn.Module` (or equivalent framework).
6. **Create the `Dockerfile`** based on an appropriate Azure ML curated base image.
7. **Pin all dependencies** in `requirements.txt` with explicit versions.
8. **Write the lab `README.md`** following the template above.
9. **Create `.amlignore`** and `.gitignore` to exclude data files and build artefacts.
10. **Create `lab.json`** with the lab metadata (see below). A GitHub Action will automatically update the root README labs table on merge to `main`.

### `lab.json` Format

Every lab must include a `lab.json` at its root. This file drives the automated labs table in the root README **and** the GitHub Pages site:

```json
{
  "name": "C++ Model Training",
  "industry": "Industry Agnostic",
  "description": "Train a linear regression model in pure C++ with Azure ML pipeline orchestration.",
  "detailedDescription": "This lab proves that native C++ model training works on the Azure ML platform...",
  "language": ["C++", "Python"],
  "useCase": ["Model Training", "Pipeline Orchestration"],
  "authors": ["ejones18"]
}
```

**Required fields:** `name`, `industry`, `description`.

**Optional fields (recommended):** `detailedDescription`, `language`, `useCase`, `authors`.

**Optional fields (external labs):** `githubPath`.

- `detailedDescription` — Longer explanation shown in the modal on the GitHub Pages site. Falls back to `description` if omitted.
- `language` — Array of programming languages used (e.g. `["Python", "C++"]`). Used for filtering on the Pages site.
- `useCase` — Array of ML use-case tags (e.g. `["Model Training", "Batch Inference"]`). Used for filtering on the Pages site.
- `authors` — Array of GitHub usernames (without `@`). Displayed on lab cards.
- `githubPath` — Full URL to the lab in an external repository. When set, overrides the default link in both the README table and the Pages site, and the lab is tagged as "External".

### Naming Conventions

| Item | Convention | Example |
|------|-----------|---------|
| Lab directory | `snake_case` | `demand_forecasting/` |
| Python files | `snake_case.py` | `preprocess_step.py` |
| Model classes | `PascalCase` | `DemandForecaster` |
| Dataset classes | `PascalCase` | `EnergyDemandDataset` |
| Azure ML environments | `kebab-case` | `demand-forecasting-env` |
| Azure ML endpoints | `kebab-case` | `demand-forecasting-batch` |
| Azure ML data assets | `kebab-case` | `demand-processed-training` |
| Experiment names | `kebab-case` | `demand-forecasting` |
| Compute clusters | lowercase short name | `gpu1`, `cpu1` |

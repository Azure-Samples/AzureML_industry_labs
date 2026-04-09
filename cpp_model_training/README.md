# 🔧 C++ Model Training on Azure ML

Train a linear regression model in **pure C++** using an Azure ML pipeline — proving that native C++ training code works on the platform.

---

## Problem Statement

Many teams have existing ML training code written in C++ — whether for performance-critical workloads, legacy codebases, or integration with C++ libraries. A common question is: **can Azure ML orchestrate C++ model training?**

The answer is **yes**. Azure ML pipelines can run any executable inside a Docker container. This lab demonstrates the pattern: compile C++ at Docker build time, invoke the binary from a pipeline step, and use MLflow for experiment tracking and model registration.

---

## Dataset

Synthetic data generated in-pipeline:

| Property | Value |
|----------|-------|
| **Samples** | 1,000 (800 train / 200 test) |
| **Features** | 3 (`x1`, `x2`, `x3`) — standard normal |
| **Target** | `y = 3x₁ + 1.5x₂ - 2x₃ + 7 + ε` |
| **Noise** | Gaussian, σ = 0.5 |
| **Format** | CSV with header row |

---

## Model

**Linear regression** via batch gradient descent, implemented in pure C++ with no external libraries.

```
y_hat = w1*x1 + w2*x2 + w3*x3 + bias
```

The C++ binary:
- Reads `train.csv` and `test.csv`
- Runs gradient descent to minimise MSE
- Writes `model_weights.json`, `test_mae.txt`, `metrics.txt`

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Learning Rate** | 0.01 |
| **Epochs** | 1,000 |
| **Loss** | Mean Squared Error |
| **Optimiser** | Batch gradient descent |

---

## Prerequisites

- Azure subscription with Contributor access
- Azure ML workspace with a **CPU** compute cluster (named `cpu1`)
- Azure CLI with ML extension (`az extension add -n ml`)
- Python 3.10+

---

## How to Run

### 1. Build and register the environment (one-time)

```bash
cd cpp_model_training
az ml environment create \
  --name cpp-training-env \
  --build-context . \
  --dockerfile-path environment/Dockerfile \
  --resource-group <rg> \
  --workspace-name <ws>
```

This installs `g++`, compiles `src/train.cpp` into `/usr/local/bin/train_cpp`, and installs Python dependencies. A `.dockerignore` ensures only the necessary files are uploaded.

### 2. Submit the pipeline

```bash
python main.py
```

### 3. Monitor in Azure ML Studio

The Studio URL is printed on submission. You'll see:
- The C++ binary's stdout (training progress, learned weights vs. true weights)
- MLflow metrics (test MAE, train MSE, individual weight values)

---

## Pipeline Steps

| Step | Script | What it does |
|------|--------|-------------|
| **Preprocess** | `pipeline/preprocess_step.py` | Generates synthetic CSV data (Python) |
| **Train** | `pipeline/train_step.py` → `train_cpp` | Invokes the compiled C++ binary, then logs metrics to MLflow (Python wrapper) |
| **Register** | `pipeline/register_model.py` | Compares against existing model, registers via MLflow if better |

---

## Outputs

| Artefact | Location | Description |
|----------|----------|-------------|
| Model weights | `model_weights.json` | Learned `w1`, `w2`, `w3`, `bias` as JSON |
| Registered model | `cpp-linear-regression` in model registry | Best model with MAE tags |
| MLflow metrics | Experiment run | test_mae, best_train_mse, individual weights |

---

## Project Structure

```
cpp_model_training/
├── main.py                      # Pipeline orchestration & job submission
├── lab.json                     # Lab metadata for CI
├── README.md                    # This file
├── .amlignore                   # Azure ML snapshot exclusions
├── .dockerignore                # Limits Docker build context upload
├── .gitignore                   # Git exclusions
├── environment/
│   ├── Dockerfile               # Compiles C++ binary + installs Python deps
│   └── requirements.txt         # Python dependencies
├── src/
│   └── train.cpp                # C++ linear regression (gradient descent)
├── data_processing/
│   └── __init__.py
├── model/
│   └── __init__.py
└── pipeline/
    ├── preprocess_step.py       # Synthetic data generation (Python)
    ├── train_step.py            # C++ binary wrapper + MLflow logging
    └── register_model.py        # Model comparison + MLflow registration
```

---

## Key Pattern: C++ in Azure ML

The approach generalises to any compiled language:

1. **Compile at Docker build time** — the Dockerfile installs the compiler, copies source, and produces a binary
2. **Invoke from a pipeline step** — a thin Python wrapper calls the binary via `subprocess.run()` and handles MLflow logging
3. **Standard I/O contract** — the binary reads from an input folder and writes to an output folder, same as any Python step

This means you can swap in C++ training for performance-critical models while keeping the rest of the Azure ML pipeline (orchestration, tracking, registration) unchanged.

---

## Tech Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| C++ | C++11 (g++) | Model training |
| Python | 3.10+ | Pipeline orchestration, data generation |
| Azure ML SDK v2 | latest | Pipeline submission |
| MLflow | 2.9.2 | Experiment tracking + model registration |
| NumPy | <2.0 | Synthetic data generation |

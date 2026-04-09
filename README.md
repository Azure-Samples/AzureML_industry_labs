# ✨ Azure ML Industry Labs

![Azure Machine Learning](https://img.shields.io/badge/Azure%20Machine%20Learning-SDK%20v2-0078D4?logo=microsoftazure&logoColor=white)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

### Hands-on Azure ML pipelines across real-world industry use-cases

A growing collection of end-to-end machine learning labs built on **Azure Machine Learning**. Each lab demonstrates a complete MLOps workflow — from data preprocessing and model training to registration and deployment — using the Azure ML SDK v2, MLflow experiment tracking, and managed endpoints.

<!-- TODO: Replace with a banner image when available -->
<!-- ![Architecture Banner](images/banner.png) -->

---

## Why Azure Machine Learning?

Building production ML systems requires more than a training script. You need reproducible pipelines, versioned data, experiment tracking, and scalable deployment — without reinventing the wheel.

**Azure Machine Learning** provides:

- 🔁 **Reproducible Pipelines** — Declarative multi-step pipelines with the `@pipeline` decorator
- 📊 **Experiment Tracking** — Integrated MLflow for metrics, parameters, and artifacts
- 📦 **Model Registry** — Versioned models with metadata tags and lineage
- 🚀 **Managed Endpoints** — Batch and real-time inference with autoscaling
- 🖥️ **Scalable Compute** — GPU/CPU clusters that spin up on demand
- 🔒 **Enterprise Security** — Managed identity, RBAC, and private networking

---

## 📚 Explore the Labs

Each lab is a self-contained project with its own pipeline, model architecture, and documentation. Click through to get started.

<!-- LABS_TABLE_START -->
| # | Lab | Industry | Description |
|---|-----|----------|-------------|
| 1 | [Azure ML Model as MCP Server](https://github.com/Azure-Samples/AI-Gateway/tree/main/labs/azure-ml-models) | Industry Agnostic | Deploy a trained ML model to Azure ML and expose it as an MCP server via APIM for Foundry agents. |
| 2 | [Conformal Energy Forecasting](conformal_energy_forecasting/) | Energy & Utilities | Forecast hourly electricity demand with calibrated prediction intervals using Conformalized Quantile Regression (CQR) deployed via Azure ML batch endpoints. |
| 3 | [C++ Model Training](cpp_model_training/) | Industry Agnostic | Train a linear regression model in pure C++ with Azure ML pipeline orchestration. |
| 4 | [R Claims Severity Forecasting](r_claims_forecasting/) | Insurance & Financial Services | Predict insurance claims severity using an R Gamma GLM deployed as an Azure ML batch endpoint. |
<!-- LABS_TABLE_END -->

> 💡 Have an idea for a new lab? Open an issue or check [CONTRIBUTING.md](CONTRIBUTING.md) to add your own!

---

## 🚀 Quick Start

### Prerequisites

- [Python 3.10+](https://www.python.org/) with a virtual environment
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) authenticated to your subscription
- [Azure ML CLI extension](https://learn.microsoft.com/azure/machine-learning/how-to-configure-cli): `az extension add -n ml`
- [Azure Subscription](https://azure.microsoft.com/free/) with Contributor access
- [VS Code](https://code.visualstudio.com/) (recommended) with the [Azure ML extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai)

### Get Started

```bash
# Clone the repository
git clone https://github.com/Azure-Samples/AzureML_industry_labs.git
cd AzureML_industry_labs

# Pick a lab
cd cpp_model_training

# Install local dependencies (for development / testing)
pip install -r environment/requirements.txt

# Authenticate with Azure
az login
az account set --subscription <your-subscription-id>

# Configure your Azure ML workspace (create config.json or use az ml)
# See the lab README for data setup instructions

# Submit the pipeline
python main.py
```

---

## 📂 Repository Structure

```
azureml_industry_labs/
├── README.md                          # ← You are here
├── CONTRIBUTING.md                    # Guidelines for adding new labs
├── AGENTS.md                          # Workspace context for AI coding agents
├── LICENSE                            # MIT License
├── images/                            # Shared images and diagrams
│
└── cpp_model_training/                # Lab 1: Platform — C++ Model Training
    ├── main.py                        # Pipeline orchestration & submission
    ├── README.md                      # Lab-specific documentation
    ├── environment/
    │   ├── Dockerfile                 # Custom Azure ML environment
    │   └── requirements.txt           # Python dependencies
    ├── src/
    │   └── train.cpp                  # C++ training binary source
    ├── data_processing/
    │   └── __init__.py
    ├── model/
    │   └── __init__.py
    └── pipeline/
        ├── preprocess_step.py         # Synthetic data generation
        ├── train_step.py              # C++ binary invocation + MLflow logging
        └── register_model.py          # Model registry with gating logic
```

---

## 🧪 Lab Convention

Every lab follows a consistent template to make the repo easy to navigate and contribute to. See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

| Component | Purpose |
|-----------|---------|
| `main.py` | Pipeline definition using the `@pipeline` decorator and job submission |
| `pipeline/` | Individual pipeline steps (preprocess, train, register, deploy, score) |
| `model/` | Model architecture definitions |
| `data_processing/` | Reusable dataset classes and preprocessing utilities |
| `Dockerfile` | Custom environment based on Azure ML curated images |
| `requirements.txt` | Pinned Python dependencies |
| `.amlignore` | Excludes large files from Azure ML snapshots |
| `README.md` | Lab-specific docs: problem statement, pipeline overview, instructions, outputs |

---

## 📖 Resources

- 📘 [Azure Machine Learning Documentation](https://learn.microsoft.com/azure/machine-learning/)
- 🐍 [Azure ML SDK v2 (Python)](https://learn.microsoft.com/python/api/azure-ai-ml/)
- 📓 [MLflow on Azure ML](https://learn.microsoft.com/azure/machine-learning/how-to-use-mlflow-cli-runs)
- 💡 [Azure ML Examples (Official)](https://github.com/Azure/azureml-examples)

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding new labs and making improvements.

---

## ⚠️ Disclaimer

> This software is provided for demonstration and educational purposes only. It is not intended to be relied upon for any production or clinical purpose. The creators make no representations or warranties about the completeness, accuracy, reliability, or suitability of the models or pipelines included in this repository. Always validate models thoroughly before any real-world application.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.


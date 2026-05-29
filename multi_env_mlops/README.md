# Multi-Environment MLOps for Healthcare

> **This lab is hosted externally.** View the full lab at: [Azure-Samples/multi_env_mlops](https://github.com/Azure-Samples/multi_env_mlops/tree/main)

Predict 30-day hospital readmission risk with Bicep IaC, three-environment model promotion (dev → test → prod), a shared ML Registry, and identity-based auth on Azure Machine Learning.

## Problem Statement

Hospital readmissions within 30 days are a key quality metric in healthcare — they indicate gaps in care transitions, drive significant costs, and are subject to regulatory penalties. Building a predictive model is only part of the solution: deploying it safely across isolated environments with proper governance, auditability, and network controls is where the real engineering challenge lies.

This lab focuses on the **infrastructure and operations** side of MLOps. The model itself (a Gradient Boosting classifier on synthetic patient data) is intentionally simple — the value is in the end-to-end deployment pattern.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                       GitHub Actions CI/CD (OIDC)                      │
│                                                                        │
│  deploy-infra.yml                train-and-promote.yml                 │
│  ┌──────────────┐    ┌───────────┬───────────┬───────────────────┐    │
│  │ Lint+What-If │    │ Train on  │ Train on  │ Promote + Deploy  │    │
│  │  → Deploy    │    │   Dev     │   Test    │    to Prod        │    │
│  └──────────────┘    └───────────┴───────────┴───────────────────┘    │
└────────────────────────────────────────────────────────────────────────┘
        │                   │             │                  │
        ▼                   ▼             ▼                  ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│rg-readmit-   │ │rg-readmit-   │ │rg-readmit-   │ │rg-readmit-prod  │
│  shared      │ │  dev         │ │  test        │ │                  │
│              │ │              │ │              │ │ ML Workspace     │
│ ML Registry  │ │ ML Workspace │ │ ML Workspace │ │ └─ Online        │
│ (model       │ │ ├─ Compute   │ │ ├─ Compute   │ │    Endpoint     │
│  promotion)  │ │ ├─ Pipeline  │ │ ├─ Pipeline  │ │ (inference only)│
│              │ │ └─ Register  │ │ └─ Register  │ │                  │
└──────┬───────┘ └──────────────┘ └──────┬───────┘ └────────▲─────────┘
       │                                 │                   │
       │              az ml model share  │                   │
       │◄────────────────────────────────┘                   │
       │                                                     │
       └─────────────────────────────────────────────────────┘
                    azureml://registries/...
```

### Environment Roles

| Environment | Resource Group | Purpose |
|-------------|---------------|---------|
| **Shared** | `rg-readmit-shared` | ML Registry only — acts as the central artefact store for cross-workspace model promotion |
| **Dev** | `rg-readmit-dev` | Validate pipeline end-to-end on synthetic data. Fast iteration, no approval gates. |
| **Test** | `rg-readmit-test` | Re-train and validate model quality on real (or representative) data. Model registered here is the promotion candidate. |
| **Prod** | `rg-readmit-prod` | Inference only. Deploys the test-validated model from the shared registry via a managed online endpoint. |

## What's Included

| Tier | Component | Description |
|------|-----------|-------------|
| **IaC** | `infra/` | Bicep modules for ML Workspace, shared Registry, compute, RBAC (identity-based, no keys), optional managed VNet |
| **Data Science** | `data_science/` | Python 3.13 pipeline: synthetic data generation on-compute, prep, train (GBM + MLflow), evaluate with promotion gate, SDK v2 model registration |
| **Components** | `mlops/azureml/components/` | Reusable component definitions registered in the shared ML Registry — all workspaces consume the same versioned code |
| **CI/CD** | `.github/workflows/` | GitHub Actions with OIDC: infra deploy (lint → what-if → deploy) and 5-stage register → train dev → train test → deploy prod |
| **Observability** | Built-in | Log Analytics, App Insights, diagnostic settings on every workspace |

## Prerequisites

- Azure subscription with **Contributor** access
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) with the `ml` extension (`az extension add -n ml`)
- Python 3.13+ (for local data generation only)

## How to Run

### 1. Deploy Infrastructure

```bash
# Login
az login

# Create resource groups
az group create --name rg-readmit-shared --location swedencentral
az group create --name rg-readmit-dev    --location swedencentral
az group create --name rg-readmit-test   --location swedencentral
az group create --name rg-readmit-prod   --location swedencentral

# Get your user principal ID (for optional Contributor role on RG)
USER_ID=$(az ad signed-in-user show --query id -o tsv)

# Deploy shared ML Registry
az deployment group create \
  --resource-group rg-readmit-shared \
  --template-file infra/shared.bicep \
  --parameters infra/parameters/shared.bicepparam

# Capture registry ID for environment RBAC
REGISTRY_ID=$(az ml registry show \
  --name readmit-registry \
  --resource-group rg-readmit-shared \
  --query id -o tsv)

# Deploy each environment workspace (passes registry ID for cross-RG RBAC)
for ENV in dev test prod; do
  az deployment group create \
    --resource-group rg-readmit-$ENV \
    --template-file infra/main.bicep \
    --parameters infra/parameters/$ENV.bicepparam \
    --parameters userPrincipalId=$USER_ID \
    --parameters mlRegistryId=$REGISTRY_ID

  # Switch datastores to identity-based auth
  az ml workspace update \
    --name readmit-$ENV-ws \
    --resource-group rg-readmit-$ENV \
    --system-datastores-auth-mode identity
done
```

### 2. Register Components to the Shared Registry

```bash
# Register the training environment in the shared registry
az ml environment create --file mlops/azureml/train/train-env.yml \
  --registry-name readmit-registry

# Register all pipeline components in the shared registry
for comp in generate_data prep train evaluate register; do
  az ml component create --file mlops/azureml/components/$comp.yml \
    --registry-name readmit-registry
done
```

### 3. Run Pipeline (Dev)

The pipeline generates its own synthetic data on the compute cluster — no manual upload needed.

```bash
# Submit the pipeline (components are pulled from the shared registry)
az ml job create --file mlops/azureml/train/pipeline.yml \
  --resource-group rg-readmit-dev --workspace-name readmit-dev-ws --stream
```

### 4. Promote and Deploy

The `train-and-promote.yml` workflow automates the full promotion flow:

1. **Register** — push environment + components to the shared ML Registry
2. **Dev** — train on synthetic data, validate the pipeline works end-to-end
3. **Test** — re-train on real data, validate model quality passes the AUC gate
4. **Prod** — `az ml model share` promotes the model from test to the registry, then deploys a managed online endpoint

Each training/deploy stage requires approval via GitHub Environments.

## Managed VNet Option

All environments default to **public networking**. To enable managed VNet isolation for any environment, set `enableManagedVnet = true` in the corresponding parameter file:

```
param enableManagedVnet = true
```

When enabled, Azure ML creates private endpoints for Storage, Key Vault, ACR, and App Insights automatically. The ACR is upgraded to Premium (required for private link). No custom VNet or NSG configuration is needed.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **3 environments + shared registry** | Dev for pipeline validation, test for model quality, prod for inference — clear separation of concerns with a shared registry as the handoff mechanism |
| **Registry-based components** | Pipeline components are registered in the shared registry and consumed by all workspaces — guarantees dev, test, and prod run identical code |
| **Identity-based auth only** | `allowSharedKeyAccess: false` on storage, `enableRbacAuthorization: true` on Key Vault, `auth_mode: aad_token` on endpoints — no keys or secrets anywhere |
| **Synthetic data on-compute** | The `generate_data` component creates data on the compute cluster, eliminating manual uploads and RBAC friction with identity-based storage |
| **SDK v2 model registration** | `register.py` uses `azure-ai-ml` SDK instead of `mlflow.register_model` to avoid a known `azureml-mlflow` + `mlflow≥2.15` incompatibility with artifact operations |
| **Bicep over Terraform** | Azure-native, first-class AML support, no state file to manage |
| **OIDC federation** | GitHub Actions authenticate via federated identity — no stored secrets |
| **GradientBoostingClassifier** | Simple, interpretable, works well on tabular healthcare data — keeps focus on infra |

## Project Structure

```
multi_env_mlops/
├── main.py                              # Local pipeline submission (SDK v2)
├── requirements.txt                     # Local dev dependencies
├── lab.json
├── README.md
├── .gitignore / .amlignore
├── infra/
│   ├── main.bicep                       # Per-environment orchestrator
│   ├── shared.bicep                     # Shared ML Registry orchestrator
│   ├── parameters/
│   │   ├── shared.bicepparam            # Registry params
│   │   ├── dev.bicepparam
│   │   ├── test.bicepparam
│   │   └── prod.bicepparam
│   └── modules/
│       ├── ml-workspace.bicep           # Workspace + Storage, KV, ACR, AppInsights, Logs
│       ├── ml-registry.bicep            # Shared ML Registry
│       ├── ml-compute.bicep             # Compute cluster (SystemAssigned MI)
│       ├── role-assignments.bicep       # RBAC (workspace MI, compute MI, user)
│       └── registry-role.bicep          # Cross-RG registry RBAC (deployed to shared RG)
├── data_science/
│   ├── config.py                        # Shared column definitions
│   ├── environment/
│   │   └── train-conda.yml              # Python 3.13 conda env
│   └── src/
│       ├── generate_data.py             # Synthetic patient data generator
│       ├── prep.py                      # Clean, one-hot encode, split
│       ├── train.py                     # GBM with MLflow tracking
│       ├── evaluate.py                  # Test metrics + AUC promotion gate
│       └── register.py                  # Conditional registration (SDK v2)
├── mlops/
│   └── azureml/
│       ├── components/
│       │   ├── generate_data.yml         # Generate data component (registered in shared registry)
│       │   ├── prep.yml                 # Prep component (registered in shared registry)
│       │   ├── train.yml                # Train component
│       │   ├── evaluate.yml             # Evaluate component
│       │   └── register.yml             # Register component
│       ├── train/
│       │   ├── pipeline.yml             # 5-step pipeline (references registry components)
│       │   └── train-env.yml            # Environment spec (registered in shared registry)
│       └── deploy/
│           └── online/
│               ├── online-endpoint.yml  # AAD-token auth endpoint
│               └── online-deployment.yml# Blue deployment from registry
└── data/
    └── sample-request.json              # Example inference payload

# GitHub Actions workflows live at the repo root:
# .github/workflows/multi-env-deploy-infra.yml
# .github/workflows/multi-env-train-and-promote.yml
```

## Tech Stack

| Technology | Version |
|------------|---------|
| Python | 3.13 |
| scikit-learn | ≥1.5 |
| MLflow | ≥2.15 |
| Azure ML SDK | v2 (`azure-ai-ml` ≥1.20) |
| Bicep | Latest |
| GitHub Actions | v4 actions, OIDC auth |
| Azure Region | Sweden Central |

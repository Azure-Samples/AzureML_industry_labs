# Brain Tumour Classification — Azure ML Pipeline

An end-to-end MLOps pipeline for brain tumour MRI classification, deployed on Azure Machine Learning. The pipeline preprocesses raw MRI images, fine-tunes a ResNet18 model, registers the best model, and deploys it to a batch endpoint — with incremental data change detection so only new/changed data triggers reprocessing and retraining. The data can be downloaded from Kaggle: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri, and uploaded to Azure Blob Storage

## Classification Classes

| Index | Class              |
|-------|--------------------|
| 0     | Glioma tumour      |
| 1     | Meningioma tumour  |
| 2     | No tumour          |
| 3     | Pituitary tumour   |

## Project Structure

```
brain_tumour_classification/
├── config.py                    # Centralised configuration (edit this)
├── main.py                      # Pipeline orchestrator (run locally)
├── Dockerfile                   # Azure ML environment image
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── .gitignore
├── .amlignore
├── model/
│   ├── __init__.py
│   └── cnn.py                   # ResNet18 fine-tuned architecture
├── data_processing/
│   ├── __init__.py
│   └── preprocess.py            # Dataset class for .pt tensors
└── pipeline/
    ├── __init__.py
    ├── preprocess_step.py       # Step 1: Raw images → .pt tensors
    ├── train_step.py            # Step 2: Train ResNet18 + MLflow logging
    ├── register_model.py        # Step 3: Register model if improved
    ├── deploy_endpoint.py       # Step 4: Deploy to batch endpoint
    └── score.py                 # Batch scoring script
```

## Pipeline Flow

```
Raw images (datastore)
    │
    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Preprocess   │───▶│    Train     │───▶│   Register   │───▶│   Deploy     │
│  (images→.pt) │    │  (ResNet18)  │    │  (if better) │    │  (batch ep.) │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

Each step is change-aware:
- **Preprocess** compares current raw images against a stored manifest; only processes new/changed files.
- **Train** only runs if training data changed; logs metrics to MLflow.
- **Register** only registers if the new model beats the existing one on validation accuracy.
- **Deploy** only redeploys if a new model was registered or testing data changed.

## Prerequisites

- An Azure subscription with an Azure ML workspace
- A GPU compute cluster provisioned in the workspace
- A managed identity with access to the workspace (used by pipeline steps)
- Raw MRI data uploaded to the workspace's default datastore under `brain_tumour_data/` with the structure:
  ```
  brain_tumour_data/
  ├── Training/
  │   ├── glioma_tumor/
  │   ├── meningioma_tumor/
  │   ├── no_tumor/
  │   └── pituitary_tumor/
  └── Testing/
      ├── glioma_tumor/
      ├── meningioma_tumor/
      ├── no_tumor/
      └── pituitary_tumor/
  ```

## Setup

1. **Clone and configure:**
   ```bash
   git clone https://github.com/<your-username>/brain-tumour-classification.git
   cd brain-tumour-classification
   cp .env.example .env
   ```

2. **Fill in `.env`** with your Azure details (subscription ID, resource group, workspace name, managed identity client ID).

3. **Create the Azure ML environment** (one-time):
   ```bash
   az ml environment create \
     --name brain-tumour-env \
     --build-context . \
     --dockerfile-path Dockerfile
   ```

4. **Create a workspace config** file at the project root (or in `~/.azureml/`):
   ```json
   {
     "subscription_id": "<your-subscription-id>",
     "resource_group": "<your-resource-group>",
     "workspace_name": "<your-workspace-name>"
   }
   ```

5. **Install dependencies locally** (for submitting the pipeline):
   ```bash
   pip install azure-ai-ml azure-identity azureml-fsspec
   ```

6. **Run the pipeline:**
   ```bash
   python main.py
   ```

## Configuration

All tuneable values live in `config.py`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `COMPUTE_CLUSTER` | `gpu1` | Azure ML compute cluster name |
| `DEFAULT_EPOCHS` | `25` | Training epochs |
| `DEFAULT_LEARNING_RATE` | `1e-5` | Base learning rate (FC head gets 10×) |
| `DEFAULT_BATCH_SIZE` | `32` | Training batch size |
| `DEFAULT_N_AUGMENTATIONS` | `3` | Augmented copies per training image |
| `DEFAULT_IMAGE_SIZE` | `512` | Input image resolution |

## Model Architecture

ResNet18 pretrained on ImageNet, with a custom classification head:

```
ResNet18 backbone (all layers trainable)
    │
    ▼
Dropout(0.4) → Linear(512, 256) → ReLU → Dropout(0.4) → Linear(256, 4)
```

Training uses differential learning rates: the FC head trains at 10× the backbone rate, with a ReduceLROnPlateau scheduler.

## Batch Inference

Once deployed, the batch endpoint accepts a folder of preprocessed `.pt` tensors and writes predictions to `predictions.csv` with columns: `filename, predicted_class, class_index`.

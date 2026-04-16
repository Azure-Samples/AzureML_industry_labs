"""
Centralised configuration for the brain tumour classification pipeline.

Copy this file or set the corresponding environment variables before running.
Values marked with <PLACEHOLDER> must be filled in for your Azure environment.
"""

import os

# ── Azure ML Workspace ───────────────────────────────────────────
# These are read from environment variables (set them in your shell,
# .env file, or Azure ML environment).
SUBSCRIPTION_ID    = os.getenv("AZURE_SUBSCRIPTION_ID", "<YOUR_SUBSCRIPTION_ID>")
RESOURCE_GROUP     = os.getenv("AZURE_RESOURCE_GROUP", "<YOUR_RESOURCE_GROUP>")
WORKSPACE_NAME     = os.getenv("AZURE_WORKSPACE_NAME", "<YOUR_WORKSPACE_NAME>")

# Managed Identity client ID (used inside pipeline steps that run on
# Azure ML compute and authenticate via ManagedIdentityCredential).
MANAGED_IDENTITY_CLIENT_ID = os.getenv(
    "AZURE_MANAGED_IDENTITY_CLIENT_ID",
    "<YOUR_MANAGED_IDENTITY_CLIENT_ID>",
)

# ── Compute & Environment ────────────────────────────────────────
COMPUTE_CLUSTER    = os.getenv("AZURE_COMPUTE_CLUSTER", "gpu1")
ENVIRONMENT_NAME   = os.getenv("AZURE_ENVIRONMENT_NAME", "brain-tumour-env@latest")

# ── Data Assets ──────────────────────────────────────────────────
TRAINING_ASSET     = "brain-tumour-processed-training"
TESTING_ASSET      = "brain-tumour-processed-testing"
RAW_DATA_PATH      = "brain_tumour_data/"          # path inside the default datastore

# ── Model ────────────────────────────────────────────────────────
MODEL_NAME         = "brain-tumour-cnn"
NUM_CLASSES        = 4
CLASS_NAMES        = sorted(["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"])

# ── Batch Endpoint ───────────────────────────────────────────────
ENDPOINT_NAME      = "brain-tumour-batch"
DEPLOYMENT_NAME    = "brain-tumour-deployment"

# ── Training Defaults ────────────────────────────────────────────
DEFAULT_EPOCHS        = 25
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_BATCH_SIZE    = 32
DEFAULT_VAL_SPLIT     = 0.2
DEFAULT_DROPOUT       = 0.4

# ── Preprocessing Defaults ───────────────────────────────────────
DEFAULT_N_AUGMENTATIONS = 3
DEFAULT_IMAGE_SIZE      = 512

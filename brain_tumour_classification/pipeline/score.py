import os
import glob
import torch
import torch.nn as nn
from torchvision import models

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLASS_NAMES, NUM_CLASSES, DEFAULT_DROPOUT


class BrainTumourCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DEFAULT_DROPOUT):
        super(BrainTumourCNN, self).__init__()
        self.model = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def init():
    global model

    model_path = os.environ.get("AZUREML_MODEL_DIR")
    if not model_path:
        raise ValueError("AZUREML_MODEL_DIR not set")

    pt_files = glob.glob(os.path.join(model_path, "**", "*.pt"), recursive=True)
    if not pt_files:
        raise FileNotFoundError(f"No .pt file found in {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m      = BrainTumourCNN(num_classes=NUM_CLASSES).to(device)
    m.load_state_dict(torch.load(pt_files[0], map_location=device, weights_only=True))
    m.eval()
    model  = m
    print(f"✅ Model loaded from {pt_files[0]}")


def run(mini_batch):
    results = []
    device  = next(model.parameters()).device

    for pt_path in mini_batch:
        try:
            data   = torch.load(pt_path, weights_only=True)
            tensor = data["tensor"].unsqueeze(0).to(device)

            with torch.no_grad():
                logits     = model(tensor)
                pred_idx   = logits.argmax(1).item()
                pred_label = CLASS_NAMES[pred_idx]

            results.append(f"{os.path.basename(pt_path)},{pred_label},{pred_idx}")
        except Exception as e:
            results.append(f"{os.path.basename(pt_path)},error,-1")
            print(f"[WARN] Failed on {pt_path}: {e}")

    return results

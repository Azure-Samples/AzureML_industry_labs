import torch
import torch.nn as nn
from torchvision import models

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_CLASSES, DEFAULT_DROPOUT


class BrainTumourCNN(nn.Module):
    """
    Fine-tuned ResNet18 for brain tumour classification.
    All layers are trainable — pretrained ImageNet weights provide
    a strong starting point, fine-tuned end-to-end on brain tumour data.
    Input:  (batch, 3, 512, 512)
    Output: (batch, num_classes)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DEFAULT_DROPOUT):
        super(BrainTumourCNN, self).__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

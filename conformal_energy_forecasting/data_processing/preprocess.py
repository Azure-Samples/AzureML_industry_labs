import os
import glob
import torch
from torch.utils.data import Dataset


FEATURE_NAMES = [
    "hour", "day_of_week", "month", "is_weekend",
    "temperature",
    "demand_lag_1", "demand_lag_24", "demand_lag_168",
    "demand_roll_mean_24", "demand_roll_mean_168",
]


class EnergyDemandDataset(Dataset):
    """
    Loads preprocessed .pt tensors for energy demand forecasting.
    Each .pt file contains {"features": Tensor, "target": Tensor, "timestamp": str}.
    """

    def __init__(self, root_dir: str, split: str):
        assert split in ("train", "calibration", "test"), \
            f"split must be 'train', 'calibration', or 'test', got {split}"

        self.samples = []
        split_dir = os.path.join(root_dir, split)

        if not os.path.exists(split_dir):
            print(f"[WARN] Split folder not found: {split_dir}")
        else:
            self.samples = sorted(glob.glob(os.path.join(split_dir, "*.pt")))

        print(f"\n── {split} dataset ────────────────────────")
        print(f"  Loaded: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(self.samples[idx], weights_only=True)
        return data["features"], data["target"]

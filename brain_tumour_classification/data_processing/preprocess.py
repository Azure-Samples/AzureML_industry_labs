import os
import glob
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLASS_NAMES

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}


class BrainTumourDataset(Dataset):
    """
    Loads preprocessed .pt tensors from the processed data asset.
    Each .pt file contains {"tensor": ..., "label": ...}.
    """

    def __init__(self, root_dir: str, split: str):
        assert split in ("Training", "Testing"), \
            f"split must be 'Training' or 'Testing', got {split}"

        self.samples = []
        split_dir    = os.path.join(root_dir, split)

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"[WARN] Class folder not found: {class_dir}")
                continue

            for pt_path in glob.glob(os.path.join(class_dir, "*.pt")):
                self.samples.append(pt_path)

        print(f"\n── {split} dataset ────────────────────────")
        print(f"  Loaded: {len(self.samples)}")
        for cls, idx in CLASS_TO_IDX.items():
            count = sum(
                1 for p in self.samples
                if os.path.basename(os.path.dirname(p)) == cls
            )
            print(f"    {cls}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(self.samples[idx])
        return data["tensor"], data["label"]

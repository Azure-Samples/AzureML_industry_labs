"""
Claims Severity Dataset Helper

Utility class for loading the preprocessed CSV splits produced by
preprocess_step.R. Not used at inference time — provided for local
exploration and testing.
"""
import os
import glob
import pandas as pd


FEATURE_NAMES = [
    "age", "gender", "vehicle_age", "vehicle_value", "region",
    "credit_score", "n_prior_claims", "coverage_type", "policy_tenure",
]


class ClaimsSeverityDataset:
    """
    Loads preprocessed CSV files for insurance claims severity modelling.
    Each split directory contains a claims.csv with policies that had claims.
    """

    def __init__(self, root_dir: str, split: str):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got {split}"

        split_dir = os.path.join(root_dir, split)
        claims_path = os.path.join(split_dir, "claims.csv")

        if not os.path.exists(claims_path):
            print(f"[WARN] Claims file not found: {claims_path}")
            self.data = pd.DataFrame()
        else:
            self.data = pd.read_csv(claims_path)

        print(f"\n── {split} dataset ────────────────────────")
        print(f"  Loaded: {len(self.data)} claims")

    def __len__(self):
        return len(self.data)

    def get_dataframe(self):
        return self.data

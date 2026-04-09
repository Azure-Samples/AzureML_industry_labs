"""
Generate a sample inference payload for the r-claims-severity-batch endpoint.

Creates a CSV file of synthetic insurance policies for batch scoring.
The generated data matches the schema expected by the R Gamma GLM model.

Usage:
    python generate_sample_payload.py

    python generate_sample_payload.py --n_policies 100 --output-dir my_payload
"""

import os
import csv
import math
import argparse
import random

parser = argparse.ArgumentParser(description="Generate sample inference payload")
parser.add_argument(
    "--n_policies",
    type=int,
    default=50,
    help="Number of sample policies to generate (default: 50)",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="sample_payload",
    help="Output directory for CSV file (default: sample_payload/)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=99,
    help="Random seed (default: 99)",
)
args = parser.parse_args()

random.seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)

# ── Generate sample policies ─────────────────────────────────────
REGIONS = ["urban", "suburban", "rural"]
GENDERS = ["M", "F"]
COVERAGE_TYPES = ["basic", "standard", "premium"]

columns = [
    "policy_start", "age", "gender", "vehicle_age", "vehicle_value",
    "region", "credit_score", "n_prior_claims", "coverage_type",
    "policy_tenure", "has_claim", "claim_amount",
    "region_urban", "region_suburban", "gender_M",
    "coverage_standard", "coverage_premium",
    "age_squared", "log_vehicle_value", "vehicle_age_sq",
]

rows = []
print(f"\nGenerating {args.n_policies} sample policies...\n")

for i in range(args.n_policies):
    age = random.randint(18, 80)
    gender = random.choice(GENDERS)
    vehicle_age = random.randint(0, 20)
    vehicle_value = round(random.uniform(5000, 80000), 2)
    region = random.choices(REGIONS, weights=[0.5, 0.3, 0.2])[0]
    credit_score = max(300, min(850, int(random.gauss(700, 80))))
    n_prior_claims = min(random.choices(range(6), weights=[60, 25, 10, 3, 1, 1])[0], 5)
    coverage_type = random.choices(COVERAGE_TYPES, weights=[0.3, 0.5, 0.2])[0]
    policy_tenure = random.randint(0, 15)

    row = {
        "policy_start": "2024-07-15",
        "age": age,
        "gender": gender,
        "vehicle_age": vehicle_age,
        "vehicle_value": vehicle_value,
        "region": region,
        "credit_score": credit_score,
        "n_prior_claims": n_prior_claims,
        "coverage_type": coverage_type,
        "policy_tenure": policy_tenure,
        "has_claim": 1,
        "claim_amount": 0,  # unknown — to be predicted
        "region_urban": 1 if region == "urban" else 0,
        "region_suburban": 1 if region == "suburban" else 0,
        "gender_M": 1 if gender == "M" else 0,
        "coverage_standard": 1 if coverage_type == "standard" else 0,
        "coverage_premium": 1 if coverage_type == "premium" else 0,
        "age_squared": age ** 2,
        "log_vehicle_value": round(math.log(vehicle_value), 4),
        "vehicle_age_sq": vehicle_age ** 2,
    }
    rows.append(row)

    if i < 5:
        print(f"  Policy {i+1}: age={age}, {gender}, vehicle_age={vehicle_age}, "
              f"${vehicle_value:,.0f}, {region}, credit={credit_score}, "
              f"{coverage_type}, tenure={policy_tenure}yr")

if args.n_policies > 5:
    print(f"  ... ({args.n_policies - 5} more)")

# ── Write CSV ────────────────────────────────────────────────────
output_path = os.path.join(args.output_dir, "data.csv")
with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ Saved {args.n_policies} policies to {output_path}")
print(f"\nNext steps:")
print(f"  1. Upload:  az ml data create --name claims-inference-input --type uri_folder --path {args.output_dir}/")
print(f"  2. Invoke:  az ml batch-endpoint invoke --name r-claims-severity-batch --input azureml:claims-inference-input:1")
print(f"     (Replace :1 with the version number returned by the data create command)")

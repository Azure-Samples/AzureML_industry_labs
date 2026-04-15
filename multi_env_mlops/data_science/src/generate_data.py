"""Generate synthetic patient data for the readmission prediction lab.

Produces a CSV with realistic distributions for hospital readmission risk
modelling. All data is synthetic — no real PHI.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_patients(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.normal(65, 15, n).clip(18, 100).astype(int)
    gender = rng.choice(["M", "F"], n, p=[0.48, 0.52])
    num_prior_admissions = rng.poisson(1.2, n)
    length_of_stay_days = rng.exponential(5, n).clip(1, 60).astype(int)
    num_diagnoses = rng.poisson(4, n).clip(1, 16)
    num_procedures = rng.poisson(1.5, n).clip(0, 10)
    num_medications = rng.poisson(8, n).clip(0, 30)
    num_lab_results = rng.poisson(12, n).clip(1, 50)
    days_since_last_admission = rng.exponential(180, n).clip(0, 3650).astype(int)
    bmi = rng.normal(28, 6, n).clip(15, 55).round(1)
    heart_rate_avg = rng.normal(80, 12, n).clip(50, 140).round(0).astype(int)
    systolic_bp_avg = rng.normal(130, 20, n).clip(80, 200).round(0).astype(int)
    hba1c = rng.normal(6.0, 1.5, n).clip(3.5, 14.0).round(1)

    admission_type = rng.choice(
        ["Emergency", "Urgent", "Elective", "Trauma"], n, p=[0.45, 0.25, 0.25, 0.05]
    )
    discharge_disposition = rng.choice(
        ["Home", "SNF", "Home_Health", "Rehab", "AMA"],
        n,
        p=[0.55, 0.15, 0.15, 0.10, 0.05],
    )
    primary_diagnosis_group = rng.choice(
        ["Circulatory", "Respiratory", "Digestive", "Musculoskeletal", "Endocrine", "Injury", "Other"],
        n,
        p=[0.25, 0.15, 0.12, 0.10, 0.13, 0.10, 0.15],
    )
    payer_code = rng.choice(
        ["Medicare", "Medicaid", "Private", "Self_Pay", "Other"],
        n,
        p=[0.45, 0.15, 0.25, 0.10, 0.05],
    )

    # --- Readmission label (logistic model with realistic coefficients) ---
    logit = (
        -2.5
        + 0.015 * age
        + 0.25 * num_prior_admissions
        + 0.03 * length_of_stay_days
        + 0.08 * num_diagnoses
        + 0.05 * num_procedures
        + 0.02 * num_medications
        - 0.002 * days_since_last_admission
        + 0.01 * bmi
        + 0.005 * heart_rate_avg
        + 0.003 * systolic_bp_avg
        + 0.1 * hba1c
        + 0.3 * (admission_type == "Emergency").astype(float)
        + 0.2 * (discharge_disposition == "SNF").astype(float)
        - 0.15 * (discharge_disposition == "Home").astype(float)
    )
    prob = 1 / (1 + np.exp(-logit))
    readmitted_30d = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "num_prior_admissions": num_prior_admissions,
            "length_of_stay_days": length_of_stay_days,
            "num_diagnoses": num_diagnoses,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "num_lab_results": num_lab_results,
            "days_since_last_admission": days_since_last_admission,
            "bmi": bmi,
            "heart_rate_avg": heart_rate_avg,
            "systolic_bp_avg": systolic_bp_avg,
            "hba1c": hba1c,
            "admission_type": admission_type,
            "discharge_disposition": discharge_disposition,
            "primary_diagnosis_group": primary_diagnosis_group,
            "payer_code": payer_code,
            "readmitted_30d": readmitted_30d,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic patient data")
    parser.add_argument("--output", type=str, default="data/patients.csv",
                        help="Output file path for generated CSV.")
    parser.add_argument("--num_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_patients(args.num_samples, seed=args.seed)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records -> {output_path}")
    print(f"Readmission rate: {df['readmitted_30d'].mean():.1%}")


if __name__ == "__main__":
    main()

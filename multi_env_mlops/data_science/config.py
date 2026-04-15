"""Shared configuration for the patient readmission prediction pipeline."""

TARGET_COL = "readmitted_30d"

NUMERIC_COLS = [
    "age",
    "num_prior_admissions",
    "length_of_stay_days",
    "num_diagnoses",
    "num_procedures",
    "num_medications",
    "num_lab_results",
    "days_since_last_admission",
    "bmi",
    "heart_rate_avg",
    "systolic_bp_avg",
    "hba1c",
]

CATEGORICAL_COLS = [
    "gender",
    "admission_type",
    "discharge_disposition",
    "primary_diagnosis_group",
    "payer_code",
]

ALL_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

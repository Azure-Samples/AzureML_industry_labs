#!/usr/bin/env Rscript
# ── Batch scoring: predict claims severity using saved Gamma GLM ─
#
# Called by score.py for each mini-batch CSV file.
# Loads the saved .rds model, predicts on new data, and writes results.

suppressPackageStartupMessages({
  library(argparse)
})

parser <- ArgumentParser(description = "Predict claims severity with saved R model")
parser$add_argument("--model_path", type = "character", required = TRUE,
                    help = "Path to the saved model.rds file")
parser$add_argument("--input_csv",  type = "character", required = TRUE,
                    help = "Path to input CSV file with policy data")
parser$add_argument("--output_csv", type = "character", required = TRUE,
                    help = "Path to write predictions CSV")
args <- parser$parse_args()

# ── Load model ───────────────────────────────────────────────────
model <- readRDS(args$model_path)

# ── Load input data ──────────────────────────────────────────────
input_df <- read.csv(args$input_csv, stringsAsFactors = FALSE)

# Ensure factor columns match what the model expects
for (col in c("region", "gender", "coverage_type")) {
  if (col %in% names(input_df)) {
    input_df[[col]] <- as.factor(input_df[[col]])
  }
}

# ── Predict ──────────────────────────────────────────────────────
# type = "response" returns predictions on the original scale (dollars)
# type = "link" would return on the log scale
predicted_severity <- predict(model, newdata = input_df, type = "response")

# Standard errors on the link (log) scale for confidence intervals
se_link <- predict(model, newdata = input_df, type = "link", se.fit = TRUE)

# Approximate 90% prediction interval on the response scale
# Using the Gamma dispersion parameter and link SE
z_90 <- qnorm(0.95)  # 1.645 for 90% interval
log_pred  <- se_link$fit
log_se    <- se_link$se.fit
lower_90  <- exp(log_pred - z_90 * log_se)
upper_90  <- exp(log_pred + z_90 * log_se)

# ── Build output ─────────────────────────────────────────────────
output_df <- data.frame(
  predicted_severity = round(predicted_severity, 2),
  lower_90           = round(lower_90, 2),
  upper_90           = round(upper_90, 2),
  interval_width     = round(upper_90 - lower_90, 2)
)

# Include identifying columns if present
id_cols <- c("policy_start", "age", "gender", "region", "coverage_type", "vehicle_value")
for (col in id_cols) {
  if (col %in% names(input_df)) {
    output_df[[col]] <- input_df[[col]]
  }
}

# Include actual claim amount if present (for evaluation)
if ("claim_amount" %in% names(input_df)) {
  output_df$actual_claim_amount <- input_df$claim_amount
}

# ── Write output ─────────────────────────────────────────────────
write.csv(output_df, args$output_csv, row.names = FALSE)
cat(sprintf("  Predicted %d records\n", nrow(output_df)))

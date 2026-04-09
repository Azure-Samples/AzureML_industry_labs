#!/usr/bin/env Rscript
# ── Train: Gamma GLM for insurance claims severity ───────────────
#
# Fits a Generalized Linear Model with Gamma family and log link
# on the claims-only subset. This is the standard actuarial approach
# for modelling claims severity (positive, right-skewed amounts).
#
# Outputs:
#   - model.rds         — serialised GLM object
#   - metrics.json      — train/val/test metrics
#   - test_mae.txt      — test MAE for model gating

suppressPackageStartupMessages({
  library(argparse)
  library(jsonlite)
})

# ── Args ─────────────────────────────────────────────────────────
parser <- ArgumentParser(description = "Train Gamma GLM for claims severity")
parser$add_argument("--processed_data", type = "character", required = TRUE)
parser$add_argument("--model_output",   type = "character", required = TRUE)
args <- parser$parse_args()

dir.create(args$model_output, recursive = TRUE, showWarnings = FALSE)

# ── Load data ────────────────────────────────────────────────────
cat("\n── Loading claims data ─────────────────────────────\n")

train_df <- read.csv(file.path(args$processed_data, "train", "claims.csv"))
val_df   <- read.csv(file.path(args$processed_data, "val",   "claims.csv"))
test_df  <- read.csv(file.path(args$processed_data, "test",  "claims.csv"))

cat(sprintf("  Train claims: %d\n", nrow(train_df)))
cat(sprintf("  Val claims:   %d\n", nrow(val_df)))
cat(sprintf("  Test claims:  %d\n", nrow(test_df)))

# Copy normalisation stats to model output for inference
file.copy(
  file.path(args$processed_data, "norm_stats.json"),
  file.path(args$model_output, "norm_stats.json")
)

# ── Ensure factors ───────────────────────────────────────────────
for (col in c("region", "gender", "coverage_type")) {
  all_levels <- unique(c(train_df[[col]], val_df[[col]], test_df[[col]]))
  train_df[[col]] <- factor(train_df[[col]], levels = all_levels)
  val_df[[col]]   <- factor(val_df[[col]],   levels = all_levels)
  test_df[[col]]  <- factor(test_df[[col]],  levels = all_levels)
}

# ── Model formula ────────────────────────────────────────────────
# Gamma GLM with log link — standard actuarial severity model
# Features: policyholder demographics, vehicle characteristics, credit, history
formula <- claim_amount ~ age + I(age^2) + gender + vehicle_age + I(vehicle_age^2) +
  log(vehicle_value) + region + credit_score + n_prior_claims +
  coverage_type + policy_tenure

# ── Fit model ────────────────────────────────────────────────────
cat("\n── Fitting Gamma GLM ──────────────────────────────\n")

model <- glm(
  formula,
  data   = train_df,
  family = Gamma(link = "log")
)

cat("\n── Model summary ──────────────────────────────────\n")
print(summary(model))

# ── Helper: compute metrics ──────────────────────────────────────
compute_metrics <- function(model, data, label) {
  pred <- predict(model, newdata = data, type = "response")
  actual <- data$claim_amount

  mae  <- mean(abs(pred - actual))
  rmse <- sqrt(mean((pred - actual)^2))
  mape <- mean(abs(pred - actual) / pmax(abs(actual), 1)) * 100

  # Deviance (Gamma)
  deviance_val <- sum(model$family$dev.resids(actual, pred, rep(1, length(actual))))

  cat(sprintf("\n  %s metrics:\n", label))
  cat(sprintf("    MAE:      $%.2f\n", mae))
  cat(sprintf("    RMSE:     $%.2f\n", rmse))
  cat(sprintf("    MAPE:     %.2f%%\n", mape))
  cat(sprintf("    Deviance: %.4f\n", deviance_val))

  list(mae = mae, rmse = rmse, mape = mape, deviance = deviance_val)
}

# ── Evaluate on all splits ───────────────────────────────────────
cat("\n── Evaluation ─────────────────────────────────────\n")
train_metrics <- compute_metrics(model, train_df, "Train")
val_metrics   <- compute_metrics(model, val_df,   "Validation")
test_metrics  <- compute_metrics(model, test_df,  "Test")

# ── Coefficient analysis ─────────────────────────────────────────
cat("\n── Top coefficients (by absolute value) ───────────\n")
coefs <- coef(model)
coefs_sorted <- sort(abs(coefs), decreasing = TRUE)
for (i in seq_len(min(10, length(coefs_sorted)))) {
  name <- names(coefs_sorted)[i]
  val  <- coefs[name]
  cat(sprintf("  %-30s %+.6f\n", name, val))
}

# ── Dispersion parameter ────────────────────────────────────────
disp <- summary(model)$dispersion
cat(sprintf("\n  Dispersion parameter (phi): %.6f\n", disp))
cat(sprintf("  Gamma shape (1/phi):        %.4f\n", 1 / disp))

# ── Save model ───────────────────────────────────────────────────
model_path <- file.path(args$model_output, "model.rds")
saveRDS(model, model_path)
cat(sprintf("\n✅ Model saved to %s\n", model_path))

# Save the formula as text for reference
writeLines(
  deparse(formula, width.cutoff = 500),
  file.path(args$model_output, "formula.txt")
)

# ── Save metrics ─────────────────────────────────────────────────
metrics <- list(
  train      = train_metrics,
  validation = val_metrics,
  test       = test_metrics,
  dispersion = disp,
  gamma_shape = 1 / disp,
  n_train    = nrow(train_df),
  n_val      = nrow(val_df),
  n_test     = nrow(test_df),
  aic        = AIC(model),
  bic        = BIC(model),
  n_coefficients = length(coefs)
)

write_json(metrics, file.path(args$model_output, "metrics.json"), auto_unbox = TRUE)

# Write test MAE for model gating (register step reads this)
writeLines(
  as.character(round(test_metrics$mae, 6)),
  file.path(args$model_output, "test_mae.txt")
)

# Write val loss (deviance) for comparison
writeLines(
  as.character(round(val_metrics$deviance, 6)),
  file.path(args$model_output, "metrics.txt")
)

cat("\n── Training complete ───────────────────────────────\n")
cat(sprintf("  Test MAE:  $%.2f\n", test_metrics$mae))
cat(sprintf("  Test RMSE: $%.2f\n", test_metrics$rmse))
cat(sprintf("  Test MAPE: %.2f%%\n", test_metrics$mape))
cat(sprintf("  AIC:       %.2f\n", AIC(model)))
cat(sprintf("  BIC:       %.2f\n", BIC(model)))

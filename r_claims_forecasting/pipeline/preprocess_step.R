#!/usr/bin/env Rscript
# ── Preprocess: generate synthetic insurance claims data ─────────
#
# Generates a realistic synthetic dataset of motor insurance policies
# with claims severity data, splits chronologically, and saves as CSV.

suppressPackageStartupMessages({
  library(argparse)
  library(jsonlite)
})

# ── Args ─────────────────────────────────────────────────────────
parser <- ArgumentParser(description = "Generate synthetic insurance claims data")
parser$add_argument("--processed_data", type = "character", required = TRUE)
parser$add_argument("--seed", type = "integer", default = 42L)
parser$add_argument("--n_policies", type = "integer", default = 50000L)
args <- parser$parse_args()

dir.create(args$processed_data, recursive = TRUE, showWarnings = FALSE)

set.seed(args$seed)

# ── Synthetic data generation ────────────────────────────────────
cat("\n── Generating synthetic insurance data ──────────────\n")
cat(sprintf("  Policies: %d\n", args$n_policies))

n <- args$n_policies

# Policy holder features
age <- sample(18:80, n, replace = TRUE)
vehicle_age <- sample(0:20, n, replace = TRUE)
vehicle_value <- round(runif(n, min = 5000, max = 80000), 2)
region <- sample(c("urban", "suburban", "rural"), n, replace = TRUE, prob = c(0.5, 0.3, 0.2))
gender <- sample(c("M", "F"), n, replace = TRUE)
credit_score <- round(rnorm(n, mean = 700, sd = 80))
credit_score <- pmax(pmin(credit_score, 850), 300)
n_prior_claims <- rpois(n, lambda = 0.5)
coverage_type <- sample(c("basic", "standard", "premium"), n, replace = TRUE, prob = c(0.3, 0.5, 0.2))
policy_tenure <- sample(0:15, n, replace = TRUE)

# Generate claims indicator (not all policies have claims)
# Higher risk for young drivers, old vehicles, urban areas, low credit
claim_prob <- 0.15 +
  ifelse(age < 25, 0.10, 0) +
  ifelse(age > 70, 0.05, 0) +
  ifelse(vehicle_age > 10, 0.05, 0) +
  ifelse(region == "urban", 0.05, 0) +
  ifelse(credit_score < 600, 0.08, 0) +
  0.03 * n_prior_claims
claim_prob <- pmin(claim_prob, 0.95)
has_claim <- rbinom(n, 1, claim_prob)

# Claims severity (only for policies with claims)
# Modelled as Gamma distributed — standard actuarial assumption
# Shape: 2.0, Rate varies by risk factors
base_rate <- 0.002
rate <- base_rate +
  ifelse(age < 25, -0.0005, 0) +
  ifelse(vehicle_value > 40000, -0.0003, 0) +
  ifelse(coverage_type == "premium", -0.0002, 0) +
  ifelse(region == "urban", -0.0003, 0)
rate <- pmax(rate, 0.0005)

shape <- 2.0
claim_amount <- rep(0, n)
claim_idx <- which(has_claim == 1)
claim_amount[claim_idx] <- rgamma(length(claim_idx), shape = shape, rate = rate[claim_idx])
# Floor at $100 minimum claim
claim_amount[claim_idx] <- pmax(claim_amount[claim_idx], 100)

# Policy start date (uniformly spread over 3 years for temporal ordering)
days_offset <- sample(0:1094, n, replace = TRUE)
policy_start <- as.Date("2022-01-01") + days_offset

# Assemble data frame
df <- data.frame(
  policy_start = policy_start,
  age = age,
  gender = gender,
  vehicle_age = vehicle_age,
  vehicle_value = vehicle_value,
  region = region,
  credit_score = credit_score,
  n_prior_claims = n_prior_claims,
  coverage_type = coverage_type,
  policy_tenure = policy_tenure,
  has_claim = has_claim,
  claim_amount = round(claim_amount, 2),
  stringsAsFactors = FALSE
)

# Sort by policy start date for chronological splitting
df <- df[order(df$policy_start), ]
rownames(df) <- NULL

cat(sprintf("  Total policies: %d\n", nrow(df)))
cat(sprintf("  Policies with claims: %d (%.1f%%)\n",
            sum(df$has_claim), 100 * mean(df$has_claim)))
cat(sprintf("  Claim amount range: $%.2f – $%.2f\n",
            min(df$claim_amount[df$has_claim == 1]),
            max(df$claim_amount[df$has_claim == 1])))
cat(sprintf("  Mean claim amount: $%.2f\n",
            mean(df$claim_amount[df$has_claim == 1])))

# ── Feature engineering ──────────────────────────────────────────
cat("\n── Engineering features ────────────────────────────\n")

# Encode categorical variables
df$region_urban    <- as.integer(df$region == "urban")
df$region_suburban <- as.integer(df$region == "suburban")
df$gender_M        <- as.integer(df$gender == "M")
df$coverage_standard <- as.integer(df$coverage_type == "standard")
df$coverage_premium  <- as.integer(df$coverage_type == "premium")

# Derived features
df$age_squared       <- df$age^2
df$log_vehicle_value <- log(df$vehicle_value)
df$vehicle_age_sq    <- df$vehicle_age^2

# Compute normalisation statistics on numeric features for inference
numeric_cols <- c("age", "vehicle_age", "vehicle_value", "credit_score",
                  "n_prior_claims", "policy_tenure", "age_squared",
                  "log_vehicle_value", "vehicle_age_sq")
norm_stats <- list(
  means = as.list(colMeans(df[, numeric_cols])),
  sds   = as.list(sapply(df[, numeric_cols], sd))
)
write_json(norm_stats, file.path(args$processed_data, "norm_stats.json"), auto_unbox = TRUE)
cat("  Saved normalisation statistics\n")

# ── Split: train (60%) / validation (20%) / test (20%) ──────────
cat("\n── Splits ─────────────────────────────────────────\n")

n_total <- nrow(df)
train_end <- floor(0.6 * n_total)
val_end   <- floor(0.8 * n_total)

train_df <- df[1:train_end, ]
val_df   <- df[(train_end + 1):val_end, ]
test_df  <- df[(val_end + 1):n_total, ]

cat(sprintf("  Train: %d policies (%d with claims)\n",
            nrow(train_df), sum(train_df$has_claim)))
cat(sprintf("  Validation: %d policies (%d with claims)\n",
            nrow(val_df), sum(val_df$has_claim)))
cat(sprintf("  Test: %d policies (%d with claims)\n",
            nrow(test_df), sum(test_df$has_claim)))

# ── Save CSVs ────────────────────────────────────────────────────
for (split_name in c("train", "val", "test")) {
  split_dir <- file.path(args$processed_data, split_name)
  dir.create(split_dir, recursive = TRUE, showWarnings = FALSE)

  split_df <- switch(split_name,
    "train" = train_df,
    "val"   = val_df,
    "test"  = test_df
  )

  write.csv(split_df, file.path(split_dir, "data.csv"), row.names = FALSE)
  cat(sprintf("  Saved %s/data.csv (%d rows)\n", split_name, nrow(split_df)))
}

# ── Save claims-only subsets for severity modelling ──────────────
cat("\n── Saving claims-only subsets ──────────────────────\n")
for (split_name in c("train", "val", "test")) {
  split_dir <- file.path(args$processed_data, split_name)

  split_df <- switch(split_name,
    "train" = train_df,
    "val"   = val_df,
    "test"  = test_df
  )

  claims_df <- split_df[split_df$has_claim == 1, ]
  write.csv(claims_df, file.path(split_dir, "claims.csv"), row.names = FALSE)
  cat(sprintf("  Saved %s/claims.csv (%d claims)\n", split_name, nrow(claims_df)))
}

cat("\n── Preprocessing complete ──────────────────────────\n")

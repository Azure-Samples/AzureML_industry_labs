# ── Gamma GLM for Claims Severity ────────────────────────────────
#
# This module defines the model specification for the actuarial claims
# severity model. The model uses a Generalized Linear Model with:
#   - Family: Gamma (positive, right-skewed claim amounts)
#   - Link:   log (ensures positive predictions)
#
# This is the standard approach in actuarial science for modelling
# the severity component of a frequency-severity decomposition.
#
# Usage:
#   source("model/gamma_glm.R")
#   model <- fit_gamma_glm(train_data)
#   preds <- predict_severity(model, new_data)

# ── Model formula ────────────────────────────────────────────────
get_formula <- function() {
  claim_amount ~ age + I(age^2) + gender + vehicle_age + I(vehicle_age^2) +
    log(vehicle_value) + region + credit_score + n_prior_claims +
    coverage_type + policy_tenure
}

# ── Fit ──────────────────────────────────────────────────────────
fit_gamma_glm <- function(data) {
  glm(
    get_formula(),
    data   = data,
    family = Gamma(link = "log")
  )
}

# ── Predict ──────────────────────────────────────────────────────
predict_severity <- function(model, newdata, interval = TRUE) {
  pred <- predict(model, newdata = newdata, type = "response")

  result <- data.frame(predicted_severity = pred)

  if (interval) {
    se_link <- predict(model, newdata = newdata, type = "link", se.fit = TRUE)
    z_90 <- qnorm(0.95)
    result$lower_90 <- exp(se_link$fit - z_90 * se_link$se.fit)
    result$upper_90 <- exp(se_link$fit + z_90 * se_link$se.fit)
  }

  result
}

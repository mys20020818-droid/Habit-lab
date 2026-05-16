suppressPackageStartupMessages({
  library(optparse)
  library(yaml)
  library(readr)
  library(dplyr)
  library(purrr)
  library(broom)
})

option_list <- list(
  make_option("--config", type = "character"),
  make_option("--components", type = "character"),
  make_option("--out", type = "character")
)
args <- parse_args(OptionParser(option_list = option_list))

cfg <- yaml::read_yaml(args$config)
components <- readr::read_csv(args$components, show_col_types = FALSE)
dir.create(dirname(args$out), recursive = TRUE, showWarnings = FALSE)

psych_vars <- cfg$models$psychopathology
scores <- c("component_general", "component_specific_rms")

`%||%` <- function(x, y) if (is.null(x)) y else x

fit_assoc <- function(df, score, psych_var) {
  needed <- c(score, psych_var, "age")
  if (!all(needed %in% names(df))) {
    return(tibble(
      score = score,
      predictor = psych_var,
      estimate = NA_real_,
      std_error = NA_real_,
      p_value = NA_real_,
      n = 0,
      model_status = "missing_columns"
    ))
  }

  model_df <- df %>% filter(!is.na(.data[[score]]), !is.na(.data[[psych_var]]), !is.na(age))
  if (nrow(model_df) < 30) {
    return(tibble(
      score = score,
      predictor = psych_var,
      estimate = NA_real_,
      std_error = NA_real_,
      p_value = NA_real_,
      n = nrow(model_df),
      model_status = "too_few_observations"
    ))
  }

  rhs <- paste(psych_var, "+ age")
  if ("sex" %in% names(model_df)) rhs <- paste(rhs, "+ sex")
  form <- as.formula(paste(score, "~", rhs))
  model <- tryCatch(lm(form, data = model_df), error = function(e) e)
  if (inherits(model, "error")) {
    return(tibble(
      score = score,
      predictor = psych_var,
      estimate = NA_real_,
      std_error = NA_real_,
      p_value = NA_real_,
      n = nrow(model_df),
      model_status = paste("lm_failed:", model$message)
    ))
  }

  broom::tidy(model) %>%
    filter(term == psych_var) %>%
    transmute(
      score = score,
      predictor = psych_var,
      estimate = estimate,
      std_error = std.error,
      p_value = p.value,
      n = nrow(model_df),
      model_status = "ok"
    )
}

if (nrow(components) == 0) {
  readr::write_csv(tibble(
    feature = character(),
    operation = character(),
    score = character(),
    predictor = character(),
    estimate = numeric(),
    std_error = numeric(),
    p_value = numeric(),
    n = integer(),
    model_status = character(),
    q_value = numeric()
  ), args$out)
  quit(save = "no")
}

results <- components %>%
  group_by(feature, operation) %>%
  group_modify(~{
    bind_rows(purrr::map(scores, function(score) {
      bind_rows(purrr::map(psych_vars, function(psych_var) {
        fit_assoc(.x, score, psych_var)
      }))
    }))
  }) %>%
  ungroup() %>%
  mutate(q_value = p.adjust(p_value, method = cfg$models$fdr_method %||% "BH"))

readr::write_csv(results, args$out)

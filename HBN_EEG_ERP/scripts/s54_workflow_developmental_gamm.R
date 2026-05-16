suppressPackageStartupMessages({
  library(optparse)
  library(yaml)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(mgcv)
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

`%||%` <- function(x, y) if (is.null(x)) y else x

if (nrow(components) == 0 || !"age" %in% names(components)) {
  readr::write_csv(tibble(
    feature = character(),
    operation = character(),
    score = character(),
    edf_age = numeric(),
    p_age = numeric(),
    r_sq = numeric(),
    n = integer(),
    model_status = character()
  ), args$out)
  quit(save = "no")
}

fit_gam_component <- function(df, score_col) {
  df <- df %>% filter(!is.na(.data[[score_col]]), !is.na(age))
  if (nrow(df) < 30 || n_distinct(df$age) < 10) {
    return(tibble(
      score = score_col,
      edf_age = NA_real_,
      p_age = NA_real_,
      r_sq = NA_real_,
      n = nrow(df),
      model_status = "too_few_observations"
    ))
  }

  k_age <- cfg$models$gamm_k_age %||% 10
  form <- as.formula(paste0(score_col, " ~ s(age, k = ", k_age, ")"))
  if ("sex" %in% names(df)) {
    form <- as.formula(paste0(score_col, " ~ s(age, k = ", k_age, ") + sex"))
  }

  model <- tryCatch(mgcv::gam(form, data = df, method = "REML"), error = function(e) e)
  if (inherits(model, "error")) {
    return(tibble(
      score = score_col,
      edf_age = NA_real_,
      p_age = NA_real_,
      r_sq = NA_real_,
      n = nrow(df),
      model_status = paste("gam_failed:", model$message)
    ))
  }

  s_table <- summary(model)$s.table
  tibble(
    score = score_col,
    edf_age = unname(s_table[1, "edf"]),
    p_age = unname(s_table[1, "p-value"]),
    r_sq = summary(model)$r.sq,
    n = nrow(df),
    model_status = "ok"
  )
}

results <- components %>%
  group_by(feature, operation) %>%
  group_modify(~bind_rows(
    fit_gam_component(.x, "component_general"),
    fit_gam_component(.x, "component_specific_rms")
  )) %>%
  ungroup()

readr::write_csv(results, args$out)

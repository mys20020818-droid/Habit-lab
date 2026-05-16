suppressPackageStartupMessages({
  library(optparse)
  library(yaml)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(lme4)
  library(broom.mixed)
})

option_list <- list(
  make_option("--config", type = "character"),
  make_option("--features", type = "character"),
  make_option("--out", type = "character")
)
args <- parse_args(OptionParser(option_list = option_list))

cfg <- yaml::read_yaml(args$config)
features <- readr::read_csv(args$features, show_col_types = FALSE)

dir.create(dirname(args$out), recursive = TRUE, showWarnings = FALSE)

if (nrow(features) == 0) {
  readr::write_csv(tibble(
    feature = character(),
    operation = character(),
    subject = character(),
    component_general = numeric(),
    component_specific_rms = numeric(),
    icc_general = numeric(),
    icc_specific = numeric(),
    model_status = character()
  ), args$out)
  quit(save = "no")
}

fit_one_feature <- function(df, key) {
  df <- df %>%
    mutate(
      subject = factor(subject),
      task = factor(task)
    )

  if (n_distinct(df$subject) < 8 || n_distinct(df$task) < 2) {
    return(tibble(
      subject = as.character(unique(df$subject)),
      component_general = NA_real_,
      component_specific_rms = NA_real_,
      icc_general = NA_real_,
      icc_specific = NA_real_,
      model_status = "too_few_subjects_or_tasks"
    ))
  }

  model <- tryCatch(
    lmer(amplitude ~ 1 + task + (1 + task | subject), data = df, REML = TRUE),
    error = function(e) e
  )

  if (inherits(model, "error")) {
    return(tibble(
      subject = as.character(unique(df$subject)),
      component_general = NA_real_,
      component_specific_rms = NA_real_,
      icc_general = NA_real_,
      icc_specific = NA_real_,
      model_status = paste("lmer_failed:", model$message)
    ))
  }

  re <- ranef(model)$subject %>%
    tibble::rownames_to_column("subject")

  slope_cols <- setdiff(colnames(re), c("subject", "(Intercept)"))
  component_table <- re %>%
    transmute(
      subject,
      component_general = .data[["(Intercept)"]],
      component_specific_rms = if (length(slope_cols) == 0) {
        NA_real_
      } else {
        sqrt(rowMeans(across(all_of(slope_cols))^2, na.rm = TRUE))
      }
    )

  vc <- as.data.frame(VarCorr(model))
  subject_var <- vc %>%
    filter(grp == "subject", var1 == "(Intercept)", is.na(var2)) %>%
    pull(vcov)
  slope_var <- vc %>%
    filter(grp == "subject", var1 != "(Intercept)", is.na(var2)) %>%
    summarise(total = sum(vcov, na.rm = TRUE)) %>%
    pull(total)
  residual_var <- vc %>% filter(grp == "Residual") %>% pull(vcov)
  total_var <- sum(subject_var, slope_var, residual_var, na.rm = TRUE)

  component_table %>%
    mutate(
      icc_general = subject_var / total_var,
      icc_specific = slope_var / total_var,
      model_status = "ok"
    )
}

components <- features %>%
  group_by(feature, operation) %>%
  group_modify(~fit_one_feature(.x, .y)) %>%
  ungroup()

phenotype_path <- cfg$paths$phenotype_file
if (!is.null(phenotype_path) && file.exists(phenotype_path)) {
  pheno <- readr::read_csv(phenotype_path, show_col_types = FALSE)
  if (!"subject" %in% names(pheno)) {
    candidate <- intersect(c("participant_id", "src_subject_id", "EID"), names(pheno))[1]
    if (!is.na(candidate)) {
      pheno <- pheno %>% rename(subject = all_of(candidate))
    }
  }
  if ("subject" %in% names(pheno)) {
    pheno <- pheno %>%
      mutate(subject = if_else(grepl("^sub-", subject), subject, paste0("sub-", subject)))
    components <- components %>% left_join(pheno, by = "subject")
  }
}

readr::write_csv(components, args$out)

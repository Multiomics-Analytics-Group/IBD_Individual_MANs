# Locate the project root (from any working directory inside the repo) and load shared paths
.root <- getwd(); while (!file.exists(file.path(.root, "code", "common", "paths.R")) && dirname(.root) != .root) .root <- dirname(.root)
source(file.path(.root, "code", "common", "paths.R"))

library(tidyverse)

# =============================================================================
# 0. Configuration
# =============================================================================

# Sets kept in the analysis (M1 + M2 and Whole graph - (M1 + M2) are excluded)
config_labels <- c(
  "whole"          = "Whole graph",
  "whole_minus_m1" = "Whole graph - M1",
  "whole_minus_m2" = "Whole graph - M2",
  "m1"             = "M1 only",
  "m2"             = "M2 only"
)
config_levels <- unname(config_labels)
keep_sets     <- names(config_labels)

# =============================================================================
# 1. Load & filter data
# =============================================================================

ablation_graphs_raw <- read_csv(file.path(RES_ABLATION_GNN, "cross_evaluation_raw_metrics_per_run_modules.csv")) %>%
  filter(train_config %in% keep_sets, eval_set %in% keep_sets)

# =============================================================================
# 2. Per-run metrics in tidy long format
# =============================================================================

raw_long <- ablation_graphs_raw %>%
  pivot_longer(
    cols          = c(internal_auc, internal_auprc, external_auc, external_auprc),
    names_to      = c("validation_type", "metric"),
    names_pattern = "(internal|external)_(auc|auprc)",
    values_to     = "score"
  ) %>%
  mutate(
    metric       = factor(if_else(metric == "auc", "AUROC", "AUPRC"),
                          levels = c("AUPRC", "AUROC")),
    train_config = factor(config_labels[train_config], levels = config_levels),
    eval_set     = config_labels[eval_set]
  )

# Fixed eval-set order (ascending internal median score), reused everywhere
eval_order <- raw_long %>%
  filter(validation_type == "internal") %>%
  group_by(eval_set) %>%
  summarise(median_score = median(score, na.rm = TRUE), .groups = "drop") %>%
  arrange(median_score) %>%
  pull(eval_set)

raw_long <- raw_long %>%
  mutate(eval_set = factor(eval_set, levels = eval_order))

# =============================================================================
# 3. Summary Table: Mean and SD across Train and Test (Eval) Sets
# =============================================================================

# Compute summary statistics (Mean and SD)
metrics_summary <- raw_long %>%
  group_by(validation_type, train_config, eval_set, metric) %>%
  summarise(
    mean_score = mean(score, na.rm = TRUE),
    sd_score   = sd(score, na.rm = TRUE),
    .groups    = "drop"
  ) %>%
  # Combine mean and sd into a formatted string "Mean ± SD"
  mutate(
    formatted = sprintf("%.3f ± %.3f", mean_score, sd_score)
  )

# Reshape into a wide comparison table (Train Config x Eval Set x Metrics)
metrics_summary_wide <- metrics_summary %>%
  select(validation_type, train_config, eval_set, metric, formatted) %>%
  pivot_wider(
    names_from  = c(validation_type, metric),
    values_from = formatted
  ) %>%
  arrange(train_config, eval_set)

# Print as a formatted markdown table and save it
print(knitr::kable(
  metrics_summary_wide,
  col_names = c(
    "Training Config", 
    "Evaluation Set", 
    "Internal AUPRC", 
    "Internal AUROC", 
    "External AUPRC", 
    "External AUROC"
  ),
  caption = "Mean ± Standard Deviation of Classification Performance across Validation Sets"
))

write_csv(metrics_summary_wide, file.path(RES_ABLATION_GNN, "cross_evaluation_summary_table_modules.csv"))

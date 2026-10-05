# Locate the project root (from any working directory inside the repo) and load shared paths
.root <- getwd(); while (!file.exists(file.path(.root, "code", "common", "paths.R")) && dirname(.root) != .root) .root <- dirname(.root)
source(file.path(.root, "code", "common", "paths.R"))

library(tidyverse)
library(ggh4x)
library(yardstick)

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

pred_cols <- c("true_label", "predicted_label", "prob", "graph_name",
               "repeat_idx", "train_config", "eval_set", "split", "dataset_origin")

# =============================================================================
# 1. Load & filter data
# =============================================================================

ablation_graphs_raw <- read_csv(file.path(RES_ABLATION_GNN, "cross_evaluation_raw_metrics_per_run_modules.csv")) %>%
  filter(train_config %in% keep_sets, eval_set %in% keep_sets)

ablation_predictions <- read_csv(file.path(RES_ABLATION_GNN, "cross_evaluation_predictions_modules.csv")) %>%
  select(all_of(pred_cols)) %>%
  filter(train_config %in% keep_sets, eval_set %in% keep_sets)

e_coli_graph_metrics <- read_csv(file.path(RES_ABLATION_GNN, "e_coli_graph_metrics.csv"))

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
# 3. Faceted boxplots (internal = HMP, external = Hall)
# =============================================================================

plot_faceted_boxplots <- function(df, val_type, title_text) {
  df %>%
    filter(validation_type == val_type) %>%
    ggplot(aes(x = eval_set, y = score, fill = metric)) +
    geom_hline(
      aes(yintercept = ifelse(metric == "AUROC", 0.5, 0.25)),
      linetype = "dashed",
      color = "grey50"
    ) +
    geom_boxplot(
      position = position_dodge(width = 0.8),
      outlier.size = 0.8,
      outlier.alpha = 0.5,
      alpha = 0.8
    ) +
    facet_grid(train_config ~ metric, scales = "free_x", space = "free_x") +
    scale_fill_manual(
      values = c("AUROC" = "#1d91c0", "AUPRC" = "#7fcdbb")
    ) +
    labs(
      title = title_text,
      x = "Evaluation set",
      y = "Performance score",
      fill = "Metric"
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
      strip.text = element_text(face = "bold"),
      strip.background = element_rect(fill = "grey92"),
      legend.position = "none",
      panel.grid = element_blank()
    )
}

internal_boxplots <- plot_faceted_boxplots(
  raw_long,
  val_type   = "internal",
  title_text = "Internal Validation Distributions Across Repeats (HMP)"
)

# Same internal-derived eval-set order
external_boxplots <- plot_faceted_boxplots(
  raw_long,
  val_type   = "external",
  title_text = "External Validation Distributions Across Repeats (Hall)"
)

print(internal_boxplots)
print(external_boxplots)

# =============================================================================
# 4. Summary Table: Mean and SD across Train and Test (Eval) Sets
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

# Print as a formatted markdown table
knitr::kable(
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
)

# Locate the project root (from any working directory inside the repo) and load shared paths
.root <- getwd(); while (!file.exists(file.path(.root, "code", "common", "paths.R")) && dirname(.root) != .root) .root <- dirname(.root)
source(file.path(.root, "code", "common", "paths.R"))

library(tidyverse)
library(scales)
library(purrr)

TRAD_PATH <- file.path(RES_TRAD_TABLES, "traditional_summary_metrics.csv")
GRAPH_PATHS <- c(
  file.path(RES_GNN_TABLES, "graph_summary_metrics.csv")
)

load_graph_metrics <- function(path) {
  label <- "timepoints_self"  # network type of the GNN runs (previously derived from the folder name)
  read_csv(path, show_col_types = FALSE) %>% mutate(graph_source = label)
}

format_gnn_metrics <- function(df, column_pattern) {
  df %>%
    mutate(across(starts_with(column_pattern), ~ {
      cleaned <- .x %>% str_remove_all("\\[|\\]|\\.\\.\\.| ") %>% str_remove_all(",$")
      purrr::map(cleaned, ~ as.numeric(unlist(str_split(.x, ","))))
    })) %>%
    unnest(cols = starts_with(column_pattern)) %>%
    mutate(node_features_list = case_when(
      node_features_list == "['x_taxonomy_onehot']" ~ "Taxonomy",
      node_features_list == "['x_mean_clr', 'x_clr_variance']" ~ "CLR (Mean+Var)",
      node_features_list == "['x_dummy']" ~ "Dummy",
      node_features_list == "['x_func_emb_BP_idf']" ~ "BP IDF",
      node_features_list == "['x_mean_clr', 'x_clr_variance', 'x_taxonomy_onehot']" ~ "CLR + Tax",
      node_features_list == "['x_mean_clr', 'x_clr_variance', 'x_func_emb_BP_idf']" ~ "CLR + BP IDF",
      node_features_list == "['x_taxonomy_onehot', 'x_func_emb_BP_idf']" ~ "Tax + BP IDF",
      node_features_list == "['x_mean_clr', 'x_clr_variance', 'x_taxonomy_onehot', 'x_func_emb_BP_idf']" ~ "CLR + Tax + BP IDF",
      TRUE ~ node_features_list
    ))
}

# Metadata fill to bridge internal/external rows
all_gnn_raw <- GRAPH_PATHS %>%
  keep(file.exists) %>%
  map_df(load_graph_metrics) %>%
  group_by(graph_source, grid_idx, node_features_list) %>%
  fill(dual_stream_aggregation, model_type, network_type, t_str, use_edge_weights, weight_type, .direction = "downup") %>%
  ungroup()

trad_raw <- read_csv(TRAD_PATH, show_col_types = FALSE) %>%
  mutate(model = case_when(
    # model == "LR" ~ "Logistic Regression", model == "RF" ~ "Random Forest",
    model == "KNN" ~ "KNN", model == "SVM" ~ "SVM", model == "MLP" ~ "MLP", TRUE ~ model
  ))

process_metrics <- function(gnn_df, trad_df, val_filter, gnn_pattern, metric_col_gnn, metric_col_trad) {
  
  # Process and clean GNN data
  g_data <- gnn_df %>%
    filter(str_detect(val_type, val_filter)) %>%
    format_gnn_metrics(column_pattern = gnn_pattern) %>%
    # Redundancy filter for single-stream
    filter(!(use_edge_weights == FALSE & dual_stream_aggregation == "subtract")) %>%
    mutate(
      weight_label = ifelse(use_edge_weights, "Edge weights", "No edge weights"),
      agg_label = case_when(
        weight_label == "No edge weights" ~ "Single-stream",
        dual_stream_aggregation == "concat" ~ "Dual-stream (concat)",
        dual_stream_aggregation == "subtract" ~ "Dual-stream (subtract)",
        TRUE ~ "Single-stream"
      ),
      graph_source = case_when(
        graph_source == "timepoints_self" ~ "Timepoints (self)",

        TRUE ~ graph_source
      )
    ) %>% 
    mutate(graph_source = factor(graph_source)) %>% 
    filter(agg_label != "Dual-stream (subtract)") # Remove single-stream duplicates

  g_stats <- g_data %>%
    group_by(graph_source, t_str, weight_label, agg_label, node_features_list) %>%
    summarise(mean_m = mean(!!sym(metric_col_gnn), na.rm = TRUE), 
              sd_m = sd(!!sym(metric_col_gnn), na.rm = TRUE), .groups = "drop")
  
  # Process Traditional Baselines
  t_stats_base <- trad_df %>%
    filter(val_type == val_filter) %>%
    group_by(model) %>%
    summarise(mean_m = mean(!!sym(metric_col_trad), na.rm = TRUE), 
              sd_m = sd(!!sym(metric_col_trad), na.rm = TRUE), .groups = "drop") %>%
    rename(node_features_list = model)

  # Distribute baselines into every existing GNN facet
  facet_structure <- g_stats %>% distinct(graph_source, weight_label, agg_label)
  t_distributed <- facet_structure %>% 
    crossing(t_stats_base) %>% 
    mutate(t_str = "Baseline", sd_m = replace_na(sd_m, 0))
  
  combined <- bind_rows(g_stats, t_distributed) %>%
    mutate(
      t_str = case_when(
        t_str == "Baseline" ~ "Baseline",
        t_str %in% c("0_3", "0.3") ~ "0.3",
        t_str %in% c("0_4", "0.4") ~ "0.4",
        t_str %in% c("0_5", "0.5") ~ "0.5",

        TRUE ~ as.character(t_str)
      ),
      t_str = factor(t_str, levels = c("Baseline", "0.3", "0.4", "0.5"))
    )
  
  rank <- combined %>% 
    filter(t_str != "Baseline") %>% 
    group_by(node_features_list) %>%
    summarise(gm = mean(mean_m, na.rm = TRUE)) %>% 
    arrange(gm)
  
  combined %>% 
    mutate(node_features_list = factor(node_features_list, 
                                     levels = c(t_stats_base$node_features_list, rank$node_features_list)))
}

plot_stability <- function(df, title, threshold_line) {
  ggplot(df, aes(x = mean_m, y = node_features_list, color = t_str)) +
    geom_vline(xintercept = threshold_line, linetype = "dashed", color = "#494949ff", linewidth = 0.5) +
    geom_errorbarh(aes(xmin = mean_m - sd_m, xmax = mean_m + sd_m), height = 0.3, 
                   position = position_dodge(width = 0.7), alpha = 0.6) +
    geom_point(position = position_dodge(width = 0.7), size = 2) +
    # Critical: No 'Traditional' label in facets; baselines live inside GNN categories
    facet_grid(graph_source ~ weight_label + agg_label) +
    scale_color_manual(values = c("0.3" = "#e41a1c", "0.4" = "#377eb8", "0.5" = "#4daf4a", "Baseline" = "black")) +
    scale_x_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1), limits = c(0, 1.1)) +
    theme_minimal() +
    theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
          panel.border = element_rect(color = "#494949ff", fill = NA, linewidth = 0.4),
          axis.text = element_text(size = 9), strip.text = element_text(size = 10.3, face = "bold"), 
          legend.position = "bottom") +
    labs(title = title, x = "Mean score", y = NULL, color = "Edge threshold")
}

data_internal_auprc <- process_metrics(all_gnn_raw, trad_raw, "internal_hmp", "raw_val_", "raw_val_auprc", "auprc")
data_internal_auroc <- process_metrics(all_gnn_raw, trad_raw, "internal_hmp", "raw_val_", "raw_val_auc", "auc")
data_external_auprc <- process_metrics(all_gnn_raw, trad_raw, "external_hall", "raw_", "raw_auprc", "auprc")
data_external_auroc <- process_metrics(all_gnn_raw, trad_raw, "external_hall", "raw_", "raw_auc", "auc")

p1 <- plot_stability(data_internal_auprc, "Internal validation: AUPRC", 0.245)
p2 <- plot_stability(data_internal_auroc, "Internal validation: AUROC", 0.50)
p3 <- plot_stability(data_external_auprc, "External validation: AUPRC", 0.245)
p4 <- plot_stability(data_external_auroc, "External validation: AUROC", 0.50)

print(p1)
print(p2)
print(p3)
print(p4)

# save plots pdf 300 dpi
save_path <- FIG_CLASSIFICATION

if (!dir.exists(save_path)) {
  dir.create(save_path, recursive = TRUE)
}

ggsave(filename = file.path(save_path, "internal_auprc_comparison.pdf"), plot = p1, width = 6, height = 5, dpi = 300)
ggsave(filename = file.path(save_path, "internal_auroc_comparison.pdf"), plot = p2, width = 6, height = 5, dpi = 300)
ggsave(filename = file.path(save_path, "external_auprc_comparison.pdf"), plot = p3, width = 6, height = 5, dpi = 300)
ggsave(filename = file.path(save_path, "external_auroc_comparison.pdf"), plot = p4, width = 6, height = 5, dpi = 300)


library(tidyverse)
library(igraph)
library(rstatix)

# =============================================================================
# 0. Configuration
# =============================================================================

PROCESSED_DATA_PATH <- file.path("data", "processed", "pyg_datasets")
IBD_DATA_PATH       <- "data/processed/IBD_CuratedMetagenomicData/IBD_studies_counts_filtered_composite.rds"
MODULES_TABLE       <- "results/tables/analyses/attributions/node_module_membership_table.csv"
SAVE_PATH           <- "results/plots/analyses/modules"

# Metrics shown in the plots (facet order follows this vector)
metric_labels <- c(
  density      = "Induced subgraph density",
  mean_weight  = "Mean edge weight",
  transitivity = "Transitivity (clustering coefficient)"
)

group_colors <- c("Non-IBD" = "#0F4C81", "IBD" = "#D90429")

# Modules to analyse: enable M2 by uncommenting (plots then facet by module)
modules_df <- read_csv(MODULES_TABLE, show_col_types = FALSE)

clusters <- list(
  "M1" = modules_df %>% filter(module == "M1") %>% pull(species_name)
  # "M2" = modules_df %>% filter(module == "M2") %>% pull(species_name)
)

# =============================================================================
# 1. Helper functions
# =============================================================================

read_gml_igraph <- function(gml_file) {
  g <- tryCatch(read_graph(gml_file, format = "gml"), error = function(e) NULL)
  if (is.null(g)) return(NULL)
  igraph::simplify(g, remove.multiple = FALSE, remove.loops = TRUE)
}

clean_node_names <- function(g) {
  if (is.null(g)) return(NULL)
  V(g)$name <- V(g)$name %>% str_replace(".*s__", "") %>% str_replace_all("_", " ")
  g
}

# Topology metrics of the subgraph induced by each cluster's species
calculate_all_clusters <- function(g, cluster_list, min_nodes = 2) {
  map_df(names(cluster_list), function(cluster_name) {
    target_nodes   <- cluster_list[[cluster_name]]
    existing_nodes <- intersect(V(g)$name, target_nodes)

    if (length(existing_nodes) < min_nodes) {
      return(tibble(
        cluster       = cluster_name,
        nodes_present = length(existing_nodes),
        density       = NA_real_,
        mean_weight   = NA_real_,
        transitivity  = NA_real_
      ))
    }

    subg    <- induced_subgraph(g, vids = V(g)[V(g)$name %in% existing_nodes])
    n_edges <- ecount(subg)
    w       <- E(subg)$weight

    mean_w  <- if (n_edges == 0 || is.null(w)) NA_real_ else mean(w, na.rm = TRUE)

    tibble(
      cluster       = cluster_name,
      nodes_present = length(existing_nodes),
      density       = edge_density(subg),
      mean_weight   = mean_w,
      transitivity  = transitivity(subg, type = "globalundirected")
    )
  })
}

process_metadata <- function(df) {
  df %>%
    group_by(subject_id) %>%
    slice_min(days_from_first_collection, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(
      condition = case_when(
        condition == "healthy"       ~ "Non-IBD",
        condition %in% c("UC", "CD") ~ condition,
        TRUE                         ~ "Other"
      ),
      condition     = factor(condition, levels = c("Non-IBD", "UC", "CD")),
      disease_group = factor(if_else(condition == "Non-IBD", "Non-IBD", "IBD"),
                             levels = c("Non-IBD", "IBD"))
    )
}

process_dataset_graphs <- function(dataset_name, cluster_list) {
  message("Processing graphs for: ", dataset_name)
  base_path <- file.path(PROCESSED_DATA_PATH,
                         paste0(dataset_name, "_timepoints_self_w_functions_onehot"), "gml")
  gml_dir <- list.dirs(base_path, recursive = FALSE) %>% str_subset("0_3")

  if (length(gml_dir) == 0) return(NULL)

  tibble(file_path = list.files(gml_dir, pattern = "\\.gml$", full.names = TRUE)) %>%
    mutate(
      id = str_remove(basename(file_path), "\\.gml$"),
      metrics = map(file_path, ~ {
        g <- read_gml_igraph(.x) %>% clean_node_names()
        if (is.null(g)) return(NULL)
        calculate_all_clusters(g, cluster_list)
      })
    ) %>%
    unnest(metrics) %>%
    select(-file_path)
}

# =============================================================================
# 2. Load data & compute per-graph module metrics
# =============================================================================

IBD_composite <- readRDS(IBD_DATA_PATH)

HMP_metadata  <- process_metadata(IBD_composite[["HMP_2019_ibdmdb"]])
Hall_metadata <- process_metadata(IBD_composite[["HallAB_2017"]])

HMP_graph_data  <- process_dataset_graphs("HMP",  clusters)
Hall_graph_data <- process_dataset_graphs("Hall", clusters)

# =============================================================================
# 3. Statistics & faceted boxplots
# =============================================================================

run_full_analysis <- function(graph_df, metadata_df, title_label) {

  final_df <- graph_df %>%
    inner_join(metadata_df, by = c("id" = "subject_id")) %>%
    mutate(disease_group = factor(disease_group, levels = c("Non-IBD", "IBD")))

  # Long format with readable metric names
  metrics_long <- final_df %>%
    pivot_longer(
      cols      = all_of(names(metric_labels)),
      names_to  = "metric",
      values_to = "value"
    ) %>%
    mutate(metric = factor(metric,
                           levels = names(metric_labels),
                           labels = unname(metric_labels)))

  # Mean and SD per group (wide: mean_nonibd, sd_nonibd, mean_ibd, sd_ibd)
  group_summary <- metrics_long %>%
    mutate(grp = if_else(disease_group == "Non-IBD", "nonibd", "ibd")) %>%
    group_by(cluster, metric, grp) %>%
    summarise(mean = mean(value, na.rm = TRUE),
              sd   = sd(value, na.rm = TRUE),
              .groups = "drop") %>%
    pivot_wider(names_from = grp, values_from = c(mean, sd),
                names_glue = "{.value}_{grp}")

  # Wilcoxon tests, BH-adjusted across all cluster x metric tests
  # (n1 = Non-IBD, n2 = IBD), joined with the group means and SDs
  stats_table <- metrics_long %>%
    group_by(cluster, metric) %>%
    wilcox_test(value ~ disease_group) %>%
    adjust_pvalue(method = "BH") %>%
    add_significance("p.adj") %>%
    left_join(group_summary, by = c("cluster", "metric")) %>%
    select(cluster, metric, n1, n2,
           mean_nonibd, sd_nonibd, mean_ibd, sd_ibd,
           statistic, p, p.adj, p.adj.signif)

  stat_labels <- stats_table %>%
    mutate(label = sprintf("BH-adj. p = %.3f (%s)", p.adj, p.adj.signif))

  # One module -> single row of metrics; several modules -> one row per module
  multi_cluster <- n_distinct(metrics_long$cluster) > 1
  facet_layer <- if (multi_cluster) {
    facet_grid(cluster ~ metric, scales = "free_y")
  } else {
    facet_wrap(~ metric, nrow = 1, scales = "free_y")
  }

  p <- ggplot(metrics_long, aes(x = disease_group, y = value, fill = disease_group)) +
    geom_jitter(aes(color = disease_group), width = 0.1, alpha = 0.7, size = 2.5, na.rm = TRUE) +
    geom_boxplot(alpha = 0.7, outlier.shape = NA, na.rm = TRUE) +
    geom_text(
      data = stat_labels,
      aes(x = 1.5, y = Inf, label = label),
      inherit.aes = FALSE, vjust = 1.4, size = 2.8
    ) +
    facet_layer +
    scale_fill_manual(values = group_colors) +
    scale_color_manual(values = group_colors) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.15))) +
    labs(
      title = paste0("Module network properties by group - ", title_label),
      x = "Condition",
      y = "Value"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid      = element_blank(),
      strip.text      = element_text(face = "bold", size = 12),
      axis.line       = element_line(color = "black"),
      axis.ticks      = element_line(color = "black"),
      axis.text       = element_text(color = "black", size = 11),
      axis.title      = element_text(size = 12),
      legend.position = "none"
    )

  list(plot = p, statistics = stats_table, data = metrics_long)
}

# Internal = HMP, external = Hall
HMP_results  <- run_full_analysis(HMP_graph_data,  HMP_metadata,  "Internal validation (HMP)")
Hall_results <- run_full_analysis(Hall_graph_data, Hall_metadata, "External validation (Hall)")

# =============================================================================
# 4. Display & save
# =============================================================================

print(HMP_results$plot)
print(Hall_results$plot)

options(tibble.width = Inf)
print(HMP_results$statistics,  n = Inf)
print(Hall_results$statistics, n = Inf)

dir.create(SAVE_PATH, recursive = TRUE, showWarnings = FALSE)
ggsave(file.path(SAVE_PATH, "module_metrics_internal_HMP.pdf"),  HMP_results$plot,  width = 11, height = 4)
ggsave(file.path(SAVE_PATH, "module_metrics_external_Hall.pdf"), Hall_results$plot, width = 11, height = 4)
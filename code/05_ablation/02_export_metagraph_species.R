# Exports the species of the importance metagraph (M) to results/ablation/metagraph_species.csv

.root <- getwd(); while (!file.exists(file.path(.root, "code", "common", "paths.R")) && dirname(.root) != .root) .root <- dirname(.root)
source(file.path(.root, "code", "common", "paths.R"))

suppressMessages({
  library(tidyverse)
  library(yardstick)
  library(tidygraph)
})

INTERP_DIR       <- RES_INTERP_AGG
OUT_FILE         <- file.path(RES_ABLATION, "metagraph_species.csv")
MIN_POOLED_AUPRC <- 0.50  # as in 04_model_interpretation.R
N_EDGES          <- 100   # as in 04_model_interpretation.R

clean_node_name <- function(x) {
  x %>%
    str_remove("^.*[s|g|f|o|c|p|k]__") %>%
    str_replace_all("_", " ")
}

# Models kept (same rule and code as 04_model_interpretation.R)
all_seeds_raw <- read_csv(file.path(INTERP_DIR, "all_seeds_attributions_raw.csv"), show_col_types = FALSE) %>%
  mutate(true_label = factor(true_label, levels = c(1, 0)))
valid_seeds <- all_seeds_raw %>%
  distinct(seed, graph_name, true_label, prob) %>%
  group_by(seed) %>%
  pr_auc(truth = true_label, prob, event_level = "first") %>%
  filter(.estimate >= MIN_POOLED_AUPRC) %>%
  pull(seed)

# Node set (all nodes with attributions in the kept models), as node_prevalence_df in 04_model_interpretation.R
node_names <- all_seeds_raw %>%
  filter(seed %in% valid_seeds) %>%
  distinct(node) %>%
  mutate(node_clean = clean_node_name(node)) %>%
  pull(node_clean) %>%
  unique()
total_study_subjects <- n_distinct(all_seeds_raw$graph_name)

# Edge ranking, as graph_edge_summary / edge_prevalence_df / top_edges_df in 04_model_interpretation.R
edges <- read_csv(file.path(INTERP_DIR, "all_seeds_edge_attributions_raw.csv"), show_col_types = FALSE) %>%
  filter(seed %in% valid_seeds) %>%
  group_by(graph_name, edge_id, source, target) %>%
  summarise(importance = mean(abs_attr_seed, na.rm = TRUE), .groups = "drop") %>%
  group_by(graph_name) %>%
  mutate(importance_norm = importance / max(importance, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(source = clean_node_name(source), target = clean_node_name(target),
         edge_id_clean = paste0(source, " -- ", target)) %>%
  group_by(edge_id_clean, source, target) %>%
  summarise(prevalence = n_distinct(graph_name) / total_study_subjects,
            mean_importance = mean(importance_norm, na.rm = TRUE),
            prevalence_weighted_importance = prevalence * mean_importance,
            .groups = "drop")

top_edges <- edges %>%
  slice_max(prevalence_weighted_importance, n = N_EDGES) %>%
  filter(source %in% node_names & target %in% node_names) %>%
  select(source, target)

# Largest connected component
metagraph <- as_tbl_graph(top_edges, directed = FALSE) %>%
  activate(nodes) %>%
  mutate(component = group_components()) %>%
  filter(component == as.numeric(names(sort(table(component), decreasing = TRUE)[1])))

species <- metagraph %>% activate(nodes) %>% as_tibble() %>% pull(name) %>% sort()
dir.create(RES_ABLATION, recursive = TRUE, showWarnings = FALSE)
write_csv(tibble(species_name = species), OUT_FILE)

message("Models kept: ", length(valid_seeds), " | edges selected: ", nrow(top_edges),
        " | metagraph species: ", length(species))
message(paste(species, collapse = "; "))

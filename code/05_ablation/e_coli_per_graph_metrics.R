library(tidyverse)
library(igraph)
library(vegan)
library(lsa)
library(ComplexHeatmap)
library(circlize)
library(grid)

# Setup Paths & Configurations
# ------------------------------------------------------------------------------
PROCESSED_DATA_PATH <- file.path("data", "processed", 
  "pyg_datasets", "HMP_timepoints_self_w_functions_onehot", "gml", "0_3")
PARENT_FOLDER <- "results/machine_learning/graph_based_models/model_interpretation/HMP_timepoints_self_t_0_3/interpretation_10_seeds"
TARGET_SPECIES <- "Escherichia coli"  # Specify the target species for analysis

# Helper Function for Taxonomy Name Cleaning
clean_node_name <- function(x) {
  x %>% 
    str_remove("^.*[s|g|f|o|c|p|k]__") %>% 
    str_replace_all("_", " ")
}

# Helper Function to Safely Perform Mann-Whitney Test
safe_wilcox_test <- function(data, formula_obj) {
  tryCatch(
    {
      res <- wilcox.test(formula_obj, data = data, alternative = "two.sided")
      as.character(round(res$p.value, 4))
    },
    error = function(e) { "N/A (insufficient samples)" }
  )
}

HMP2_scores <- read_tsv("data/external/HMP2_dysbiosis_scores.tsv", col_names = FALSE) %>% 
  rename(`External ID` = X1, dysbiosis_score = X2, dysbiosis_state = X3)


# 1. Extract Importance (Signed & Absolute)
# ------------------------------------------------------------------------------
all_seeds_raw <- read_csv(file.path(PARENT_FOLDER, "all_seeds_attributions_raw.csv"), show_col_types = FALSE) %>% 
  mutate(true_label = factor(true_label, levels = c(1, 0)))

# Filter seeds by performance threshold (AUPRC >= 0.60)
seed_auprc <- all_seeds_raw %>%
  distinct(seed, graph_name, true_label, prob) %>%
  group_by(seed) %>%
  yardstick::pr_auc(truth = true_label, prob, event_level = "first") %>%
  rename(auprc = .estimate)

valid_seeds <- seed_auprc %>% filter(auprc >= 0.60) %>% pull(seed)

ecoli_attributions <- all_seeds_raw %>%
  filter(seed %in% valid_seeds) %>%
  select(seed, graph_name, node, net_attr_seed, total_attr_seed) %>%
  mutate(node_clean = clean_node_name(node)) %>%
  filter(node_clean == TARGET_SPECIES) %>%
  group_by(graph_name) %>%
  summarise(
    importance = mean(total_attr_seed, na.rm = TRUE),
    net        = mean(net_attr_seed, na.rm = TRUE),
    .groups    = "drop"
  )

# Calculate graph-level max normalization per subject graph
graph_max_importance <- all_seeds_raw %>%
  filter(seed %in% valid_seeds) %>%
  group_by(graph_name, node) %>%
  summarise(importance = mean(total_attr_seed, na.rm = TRUE), .groups = "drop") %>%
  group_by(graph_name) %>%
  summarise(max_abs = max(importance, na.rm = TRUE), .groups = "drop")

ecoli_importance_norm <- ecoli_attributions %>%
  left_join(graph_max_importance, by = "graph_name") %>%
  mutate(
    e_coli_signed_importance   = net / max_abs,
    e_coli_absolute_importance = importance / max_abs
  ) %>%
  select(graph_name, e_coli_signed_importance, e_coli_absolute_importance)

# 2. Extract Metadata & Abundance
# ------------------------------------------------------------------------------
IBD_composite <- readRDS("data/processed/IBD_CuratedMetagenomicData/IBD_studies_counts_filtered_composite.rds")
HMP <- IBD_composite[["HMP_2019_ibdmdb"]]

HMP_timepoint_filtered <- HMP %>% 
  group_by(subject_id) %>% 
  filter(!duplicated(days_from_first_collection)) %>%
  filter(n_distinct(days_from_first_collection) >= 7) %>%
  arrange(subject_id, days_from_first_collection, id) %>%
  distinct(subject_id, days_from_first_collection, .keep_all = TRUE) %>%
  ungroup() %>% 
  left_join(HMP2_scores, by = c("id" = "External ID")) %>% 
  group_by(subject_id) %>% 
  mutate(mean_dysbiosis_score = mean(dysbiosis_score, na.rm = TRUE),
         median_dysbiosis_score = median(dysbiosis_score, na.rm = TRUE),
         pct_dysbiotic = mean(dysbiosis_state, na.rm = TRUE)) %>% 
  ungroup() %>% 
  group_by(subject_id) %>%
  mutate(number_timepoints = n_distinct(days_from_first_collection)) %>%
  ungroup() %>% 
  mutate(condition = case_when(
    disease == "healthy" ~ "Non-IBD",
    disease == "UC" ~ "UC",
    disease == "CD" ~ "CD",
    TRUE ~ condition
  )) %>% 
  mutate(condition = factor(condition, levels = c("Non-IBD", "UC", "CD"))) %>% 
  group_by(subject_id) %>%
  mutate(ab_usage_proportion = mean(antibiotics_current_use == "yes") * 100) %>% 
  ungroup()

taxa_cols <- names(HMP_timepoint_filtered)[startsWith(names(HMP_timepoint_filtered), "k__")]

HMP_counts <- HMP_timepoint_filtered %>%
  select(all_of(taxa_cols), id) %>%
  column_to_rownames("id")

HMP_clr_agg <- HMP_counts %>%
  vegan::decostand(method = "clr", pseudocount = 1) %>%
  as.data.frame() %>%
  rownames_to_column("id") %>%
  left_join(HMP_timepoint_filtered %>% select(id, subject_id), by = "id") %>%
  select(-id) %>%
  group_by(subject_id) %>%
  summarise(across(everything(), mean), .groups = "drop")

ecoli_clr <- HMP_clr_agg %>%
  pivot_longer(
    cols = starts_with("k__"),
    names_to = "node",
    values_to = "e_coli_abundance"
  ) %>%
  mutate(node_clean = clean_node_name(node)) %>%
  filter(node_clean == TARGET_SPECIES) %>%
  select(graph_name = subject_id, e_coli_abundance)

# 3. Extract Graph Topology Metrics
# ------------------------------------------------------------------------------
read_gml_igraph <- function(gml_file) {
  tryCatch(
    {
      g <- read_graph(gml_file, format = "gml")
    },
    error = function(e) { NULL }
  )
  if (!is.null(g)) {
    return(igraph::simplify(g, remove.multiple = FALSE, remove.loops = TRUE))
  }
}

gml_files <- list.files(PROCESSED_DATA_PATH, pattern = "\\.gml$", full.names = TRUE)
igraph_list <- lapply(gml_files, read_gml_igraph)
names(igraph_list) <- tools::file_path_sans_ext(basename(gml_files))

ecoli_network_metrics <- map_dfr(names(igraph_list), function(gname) {
  g <- igraph_list[[gname]]
  if (is.null(g)) return(NULL)
  
  v_names <- V(g)$name
  if (is.null(v_names)) v_names <- V(g)$label
  
  deg <- igraph::degree(g, v = V(g), mode = "all", normalized = TRUE)
  bet <- igraph::betweenness(g, v = V(g), directed = FALSE, weights = NA, normalized = TRUE)
  n_nodes <- vcount(g)
  
  data.frame(
    graph_name = gname,
    node = v_names,
    e_coli_normalized_degree = deg,
    e_coli_norm_betweenness = bet,
    num_nodes = n_nodes,
    stringsAsFactors = FALSE
  )
}) %>%
  mutate(node_clean = clean_node_name(node)) %>%
  filter(node_clean == TARGET_SPECIES) %>%
  select(graph_name, e_coli_normalized_degree, e_coli_norm_betweenness, num_nodes)

# 4. Assemble Master Dataset
# ------------------------------------------------------------------------------
all_graphs <- tibble(graph_name = unique(c(
  names(igraph_list),
  ecoli_importance_norm$graph_name,
  ecoli_clr$graph_name
)))

ecoli_metrics_table <- all_graphs %>%
  left_join(ecoli_importance_norm, by = "graph_name") %>%
  left_join(ecoli_clr, by = "graph_name") %>%
  left_join(ecoli_network_metrics, by = "graph_name") %>%
  left_join(HMP_timepoint_filtered %>% select(-any_of(taxa_cols)), by = c("graph_name" = "subject_id")) %>%
  arrange(graph_name)

plot_df <- ecoli_metrics_table %>%
  filter(!is.na(condition)) %>% 
  select(graph_name, e_coli_signed_importance, e_coli_normalized_degree, e_coli_norm_betweenness,
         e_coli_abundance, condition) %>% 
  distinct(across(everything()))

condition_colors <- c("Non-IBD" = "#0F4C81", "UC" = "#E65100", "CD" = "#D90429")


# ==============================================================================
# ANALYSIS 1: Correlations Breakdown (Non-IBD, UC, CD)
# ==============================================================================

# Correlation 1A: Signed Importance vs. Normalized Degree
ggplot(plot_df, aes(
  x = e_coli_signed_importance, 
  y = e_coli_normalized_degree, 
  color = condition
)) +
  geom_point(aes(size = e_coli_signed_importance < -0.2), alpha = 0.8) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.8, color = "grey30") +
  facet_wrap(~ condition) +
  scale_color_manual(values = condition_colors, drop = TRUE) +
  scale_size_manual(values = c(`FALSE` = 2.5, `TRUE` = 4.5), guide = "none") +
  labs(
    title = "E. coli Signed Importance vs. Normalized Degree",
    x = "Signed importance",
    y = "Normalized degree",
    color = "Condition"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    strip.text = element_text(face = "bold"),
    panel.border = element_rect(color = "black", fill = NA),
    axis.ticks = element_line(color = "black"),
    axis.text = element_text(color = "black", size = 10),
    axis.title = element_text(size = 12),
    legend.position = "none"
  )

# Correlation 1B: Signed Importance vs. Normalized Betweenness
ggplot(plot_df, aes(
  x = e_coli_signed_importance, 
  y = e_coli_norm_betweenness, 
  color = condition
)) +
  geom_point(aes(size = e_coli_signed_importance < -0.2), alpha = 0.8) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.8, color = "grey30") +
  facet_wrap(~ condition) +
  scale_color_manual(values = condition_colors, drop = TRUE) +
  scale_size_manual(values = c(`FALSE` = 2.5, `TRUE` = 4.5), guide = "none") +
  labs(
    title = "E. coli Signed Importance vs. Normalized Betweenness",
    x = "Signed importance",
    y = "Normalized betweenness",
    color = "Condition"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    strip.text = element_text(face = "bold"),
    panel.border = element_rect(color = "black", fill = NA),
    axis.ticks = element_line(color = "black"),
    axis.text = element_text(color = "black", size = 10),
    axis.title = element_text(size = 12),
    legend.position = "none"
  )


# ==============================================================================
# ANALYSIS 2: Boxplots (Negative vs. Positive Importance by Condition)
# ==============================================================================

# Define discrete importance groups
selected_graphs <- plot_df %>%
  filter(!is.na(e_coli_signed_importance)) %>%
  mutate(group = case_when(
    e_coli_signed_importance < 0 ~ "Negative",
    e_coli_signed_importance > 0 ~ "Positive",
    TRUE ~ NA_character_
  )) %>%
  filter(!is.na(group)) %>%
  distinct(graph_name, .keep_all = TRUE)

primary_edge_stats <- map_dfr(names(igraph_list), function(gname) {
  g <- igraph_list[[gname]]
  if (is.null(g)) return(NULL)
  
  v_names <- clean_node_name(V(g)$name)
  ego_idx <- which(v_names == TARGET_SPECIES)
  if (length(ego_idx) == 0) return(NULL)
  
  inc_edges <- incident(g, v = ego_idx[[1]], mode = "all")
  weights <- E(g)$weight[inc_edges]
  
  tibble(
    graph_name = gname,
    norm_degree = igraph::degree(g, v = ego_idx, mode = "all", normalized = TRUE),
    unweighted_betweenness = igraph::betweenness(g, v = ego_idx, directed = FALSE, weights = NA, normalized = TRUE),
    mean_abs_weight = mean(abs(weights), na.rm = TRUE),
    prop_positive = mean(weights > 0, na.rm = TRUE)
  )
}) %>% 
  inner_join(selected_graphs, by = "graph_name")

# Mann-Whitney U Tests per Condition
p_deg_CD      <- safe_wilcox_test(filter(primary_edge_stats, condition == "CD"), norm_degree ~ group)
p_bet_CD      <- safe_wilcox_test(filter(primary_edge_stats, condition == "CD"), unweighted_betweenness ~ group)
p_deg_UC      <- safe_wilcox_test(filter(primary_edge_stats, condition == "UC"), norm_degree ~ group)
p_bet_UC      <- safe_wilcox_test(filter(primary_edge_stats, condition == "UC"), unweighted_betweenness ~ group)
p_deg_non_ibd <- safe_wilcox_test(filter(primary_edge_stats, condition == "Non-IBD"), norm_degree ~ group)
p_bet_non_ibd <- safe_wilcox_test(filter(primary_edge_stats, condition == "Non-IBD"), unweighted_betweenness ~ group)

# Boxplot 2A: Normalized Degree
degree_plot <- primary_edge_stats %>%
  ggplot(aes(x = group, y = norm_degree, fill = group)) +
  geom_jitter(width = 0.1, alpha = 0.7, size = 2.5, aes(color = group)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  scale_fill_manual(values = c("Negative" = "#d7191c", "Positive" = "#2c7bb6")) +
  scale_color_manual(values = c("Negative" = "#d7191c", "Positive" = "#2c7bb6")) +
  theme_minimal(base_size = 12) +
  facet_wrap(~ condition, nrow = 3) +
  labs(
    title = "E. coli Normalized Degree by Importance Group",
    subtitle = paste0("Wilcoxon p: CD = ", p_deg_CD, " | UC = ", p_deg_UC, " | Non-IBD = ", p_deg_non_ibd),
    x = "E. coli node signed importance group",
    y = "Normalized degree"
  ) +
  coord_flip() +
  theme(
    panel.grid = element_blank(),
    strip.text = element_text(face = "bold", size = 14),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black"),
    axis.text = element_text(color = "black", size = 12),
    axis.title = element_text(size = 13),
    legend.position = "none"
  )

# Boxplot 2B: Unweighted Betweenness
betweenness_plot <- primary_edge_stats %>%
  ggplot(aes(x = group, y = unweighted_betweenness, fill = group)) +
  geom_jitter(width = 0.1, alpha = 0.7, size = 2.5, aes(color = group)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  scale_fill_manual(values = c("Negative" = "#d7191c", "Positive" = "#2c7bb6")) +
  scale_color_manual(values = c("Negative" = "#d7191c", "Positive" = "#2c7bb6")) +
  theme_minimal(base_size = 12) +
  facet_wrap(~ condition, nrow = 3) +
  coord_flip() +
  labs(
    title = "E. coli Unweighted Betweenness by Importance Group",
    subtitle = paste0("Wilcoxon p: CD = ", p_bet_CD, " | UC = ", p_bet_UC, " | Non-IBD = ", p_bet_non_ibd),
    x = "E. coli node signed importance group",
    y = "Normalized Betweenness"
  ) +
  theme(
    panel.grid = element_blank(),
    strip.text = element_text(face = "bold", size = 14),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black"),
    axis.text = element_text(color = "black", size = 12),
    axis.title = element_text(size = 13),
    legend.position = "none"
  )


# ==============================================================================
# ANALYSIS 3: Edge Profile Cosine Similarity to Non-IBD Baseline (UC & CD)
# ==============================================================================

# Extract all E. coli-centric edges
all_ecoli_edges <- map_dfr(names(igraph_list), function(gname) {
  g <- igraph_list[[gname]]
  if (is.null(g)) return(NULL)
  
  v_names <- clean_node_name(V(g)$name)
  ego_idx <- which(v_names == TARGET_SPECIES)
  if (length(ego_idx) == 0) return(NULL)
  
  nb_idx <- neighbors(g, v = ego_idx, mode = "all")
  if (length(nb_idx) == 0) return(NULL)
    
  edge_ids <- incident(g, v = ego_idx, mode = "all")
  weights <- edge_attr(g, "weight", index = edge_ids)
  
  tibble(
    graph_name = gname,
    neighbor_species = v_names[nb_idx],
    correlation_weight = weights
  )
})

# Pivot into wide edge matrix
edge_matrix_df <- all_ecoli_edges %>%
  pivot_wider(
    names_from = neighbor_species,
    values_from = correlation_weight,
    values_fill = 0
  )

graph_metadata <- plot_df %>%
  select(graph_name, condition, e_coli_signed_importance) %>%
  mutate(
    importance_group = case_when(
      e_coli_signed_importance > 0 ~ "Positive",
      e_coli_signed_importance < 0 ~ "Negative",
      TRUE ~ "Neutral"
    )
  )

edge_profiles <- edge_matrix_df %>%
  inner_join(graph_metadata, by = "graph_name")

non_ibd_profiles <- edge_profiles %>% filter(condition == "Non-IBD")
target_profiles  <- edge_profiles %>% filter(condition %in% c("UC", "CD") & importance_group %in% c("Positive", "Negative"))

feature_cols <- setdiff(names(edge_matrix_df), "graph_name")

# Calculate average cosine similarity of each UC/CD graph to Non-IBD graphs
disease_similarity_results <- map_dfr(1:nrow(target_profiles), function(i) {
  target_row <- target_profiles[i, ]
  target_vec <- as.numeric(target_row[feature_cols])
  
  similarities <- apply(non_ibd_profiles[feature_cols], 1, function(non_ibd_vec) {
    vec1 <- as.numeric(non_ibd_vec)
    if (all(vec1 == 0) || all(target_vec == 0)) return(0)
    lsa::cosine(target_vec, vec1)[1, 1]
  })
  
  tibble(
    graph_name = target_row$graph_name,
    importance_group = target_row$importance_group,
    condition = target_row$condition,
    mean_cosine_sim_to_non_ibd = mean(similarities, na.rm = TRUE)
  )
})

# Calculate individual Wilcoxon p-values for UC and CD
p_sim_cd <- safe_wilcox_test(filter(disease_similarity_results, condition == "CD"), mean_cosine_sim_to_non_ibd ~ importance_group)
p_sim_uc <- safe_wilcox_test(filter(disease_similarity_results, condition == "UC"), mean_cosine_sim_to_non_ibd ~ importance_group)

cossimil_plot <- ggplot(disease_similarity_results, aes(x = importance_group, y = mean_cosine_sim_to_non_ibd, fill = importance_group)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA, color = "black", width = 0.4) +
  geom_jitter(width = 0.1, alpha = 0.7, size = 2.5, aes(color = importance_group)) +
  scale_fill_manual(values = c("Negative" = "#d7191c", "Positive" = "#2c7bb6")) +
  scale_color_manual(values = c("Negative" = "#d7191c", "Positive" = "#2c7bb6")) +
  facet_wrap(~ condition, nrow = 2) +
  coord_flip() +
  labs(
    title = "Similarity of UC and CD E. coli Edges to Non-IBD Baseline Graphs",
    subtitle = paste0("Wilcoxon p-values: CD = ", p_sim_cd, " | UC = ", p_sim_uc),
    x = "E. coli node signed importance group",
    y = "Mean cosine similarity to non-IBD baseline"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid = element_blank(),
    axis.line = element_line(color = "black"),
    strip.text = element_text(face = "bold", size = 14),
    axis.ticks = element_line(color = "black"),
    axis.title = element_text(size = 13),
    axis.text = element_text(color = "black", size = 12),
    legend.position = "none"
  )

# ---- 1. Raw p-values as numbers (not rounded strings) -----------------------
safe_wilcox_p <- function(data, formula_obj) {
  tryCatch(wilcox.test(formula_obj, data = data, alternative = "two.sided")$p.value,
           error = function(e) NA_real_)
}

test_by_condition <- function(df, formula_obj, label) {
  df %>%
    nest(data = -condition) %>%
    mutate(
      analysis = label,
      p        = map_dbl(data, ~ safe_wilcox_p(.x, formula_obj))
    ) %>%
    select(analysis, condition, p)
}

# ---- 2. One table with every test, then BH across the whole family ----------
ecoli_tests <- bind_rows(
  test_by_condition(primary_edge_stats,          norm_degree ~ group,                     "Normalized degree"),
  test_by_condition(primary_edge_stats,          unweighted_betweenness ~ group,          "Normalized betweenness"),
  test_by_condition(disease_similarity_results,  mean_cosine_sim_to_non_ibd ~ importance_group, "Cosine similarity")
) %>%
  mutate(p_adj = p.adjust(p, method = "BH"))

print(ecoli_tests, n = Inf)

# ---- 3. Subtitles showing raw and adjusted p-values -------------------------
fmt_tests <- function(analysis_name) {
  ecoli_tests %>%
    filter(analysis == analysis_name) %>%
    mutate(txt = sprintf("%s: p = %.3f, BH-adj. p = %.3f", condition, p, p_adj)) %>%
    pull(txt) %>%
    paste(collapse = " | ")
}

degree_plot      <- degree_plot      + labs(subtitle = paste0("Wilcoxon ", fmt_tests("Normalized degree")))
betweenness_plot <- betweenness_plot + labs(subtitle = paste0("Wilcoxon ", fmt_tests("Normalized betweenness")))
cossimil_plot    <- cossimil_plot    + labs(subtitle = paste0("Wilcoxon ", fmt_tests("Cosine similarity")))

# ---- 4. E. coli abundance (mean CLR) by importance group --------------------
# Kept as its own test family so the BH adjustment above is unaffected
abundance_df <- selected_graphs %>%
  filter(!is.na(e_coli_abundance))

ecoli_abundance_tests <- test_by_condition(abundance_df, e_coli_abundance ~ group, "Abundance") %>%
  mutate(p_adj = p.adjust(p, method = "BH"))

print(ecoli_abundance_tests, n = Inf)

abundance_subtitle <- ecoli_abundance_tests %>%
  mutate(txt = sprintf("%s: p = %.3f, BH-adj. p = %.3f", condition, p, p_adj)) %>%
  pull(txt) %>%
  paste(collapse = " | ")

abundance_plot <- abundance_df %>%
  ggplot(aes(x = group, y = e_coli_abundance, fill = group)) +
  geom_jitter(width = 0.1, alpha = 0.7, size = 2.5, aes(color = group)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  scale_fill_manual(values = c("Negative" = "#d7191c", "Positive" = "#2c7bb6")) +
  scale_color_manual(values = c("Negative" = "#d7191c", "Positive" = "#2c7bb6")) +
  theme_minimal(base_size = 12) +
  facet_wrap(~ condition, nrow = 3) +
  coord_flip() +
  labs(
    title = "E. coli Abundance by Importance Group",
    subtitle = paste0("Wilcoxon ", abundance_subtitle),
    x = "E. coli node signed importance group",
    y = "Mean CLR abundance"
  ) +
  theme(
    panel.grid = element_blank(),
    strip.text = element_text(face = "bold", size = 14),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black"),
    axis.text = element_text(color = "black", size = 12),
    axis.title = element_text(size = 13),
    legend.position = "none"
  )

# ---- 5. Percentage of dysbiotic samples per subject by importance group -----
# Kept as its own test family so the BH adjustments above are unaffected
dysbiosis_df <- selected_graphs %>%
  left_join(
    ecoli_metrics_table %>% distinct(graph_name, pct_dysbiotic),
    by = "graph_name"
  ) %>%
  filter(!is.na(pct_dysbiotic)) %>%
  mutate(pct_dysbiotic = pct_dysbiotic * 100)

ecoli_dysbiosis_tests <- test_by_condition(dysbiosis_df, pct_dysbiotic ~ group, "Dysbiotic samples") %>%
  mutate(p_adj = p.adjust(p, method = "BH"))

print(ecoli_dysbiosis_tests, n = Inf)

dysbiosis_subtitle <- ecoli_dysbiosis_tests %>%
  mutate(txt = sprintf("%s: p = %.3f, BH-adj. p = %.3f", condition, p, p_adj)) %>%
  pull(txt) %>%
  paste(collapse = " | ")

dysbiosis_plot <- dysbiosis_df %>%
  ggplot(aes(x = group, y = pct_dysbiotic, fill = group)) +
  geom_jitter(width = 0.1, alpha = 0.7, size = 2.5, aes(color = group)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  scale_fill_manual(values = c("Negative" = "#d7191c", "Positive" = "#2c7bb6")) +
  scale_color_manual(values = c("Negative" = "#d7191c", "Positive" = "#2c7bb6")) +
  theme_minimal(base_size = 12) +
  facet_wrap(~ condition, nrow = 3) +
  coord_flip() +
  labs(
    title = "Dysbiotic Samples by E. coli Importance Group",
    subtitle = paste0("Wilcoxon ", dysbiosis_subtitle),
    x = "E. coli node signed importance group",
    y = "Dysbiotic samples per subject (%)"
  ) +
  theme(
    panel.grid = element_blank(),
    strip.text = element_text(face = "bold", size = 14),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black"),
    axis.text = element_text(color = "black", size = 12),
    axis.title = element_text(size = 13),
    legend.position = "none"
  )


# ==============================================================================
# ANALYSIS 4: Heatmap Broken Down by UC & CD Importance Groups
# ==============================================================================

# Create granular condition-importance groups
edge_profiles_grouped <- edge_profiles %>%
  mutate(
    group = case_when(
      condition == "Non-IBD" ~ "Non-IBD",
      condition == "UC" & importance_group == "Positive" ~ "UC (Positive)",
      condition == "UC" & importance_group == "Negative" ~ "UC (Negative)",
      condition == "CD" & importance_group == "Positive" ~ "CD (Positive)",
      condition == "CD" & importance_group == "Negative" ~ "CD (Negative)",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(group))

# Calculate Edge Mean Weights per detailed group
edge_summary_mean <- edge_profiles_grouped %>%
  group_by(group) %>%
  summarise(across(all_of(feature_cols), \ (x) mean(x, na.rm = TRUE)), .groups = "drop") %>%
  pivot_longer(-group, names_to = "neighbor_species", values_to = "mean_weight") %>%
  pivot_wider(names_from = group, values_from = mean_weight)

# Calculate Edge Prevalences (% subjects per group with non-zero weight)
edge_summary_prev <- edge_profiles_grouped %>%
  group_by(group) %>%
  summarise(across(all_of(feature_cols), \ (x) mean(x != 0, na.rm = TRUE) * 100), .groups = "drop") %>%
  pivot_longer(-group, names_to = "neighbor_species", values_to = "prevalence") %>%
  pivot_wider(names_from = group, values_from = prevalence)

# Identify top driver species based on maximum difference from Non-IBD baseline across all groups
top_drivers <- edge_summary_mean %>%
  mutate(
    diff_uc_neg = abs(`UC (Negative)` - `Non-IBD`),
    diff_uc_pos = abs(`UC (Positive)` - `Non-IBD`),
    diff_cd_neg = abs(`CD (Negative)` - `Non-IBD`),
    diff_cd_pos = abs(`CD (Positive)` - `Non-IBD`)
  ) %>%
  slice_max(order_by = pmax(diff_uc_neg, diff_uc_pos, diff_cd_neg, diff_cd_pos, na.rm = TRUE), n = 50) %>%
  mutate(clean_species = gsub("_", " ", neighbor_species))

# Align Prevalence Data with Top Drivers
top_drivers_prev <- top_drivers %>%
  select(neighbor_species, clean_species) %>%
  left_join(edge_summary_prev, by = "neighbor_species")

heatmap_cols <- c("Non-IBD", "UC (Positive)", "UC (Negative)", "CD (Positive)", "CD (Negative)")

# Construct Heatmap Matrices
mat_weight <- top_drivers %>%
  select(all_of(heatmap_cols)) %>%
  as.matrix()
rownames(mat_weight) <- top_drivers$clean_species

mat_prev <- top_drivers_prev %>%
  select(all_of(heatmap_cols)) %>%
  as.matrix()
rownames(mat_prev) <- top_drivers_prev$clean_species

# Define Color Scale Parameters
min_val <- min(mat_weight, na.rm = TRUE)
max_val <- max(mat_weight, na.rm = TRUE)
max_abs <- max(abs(c(min_val, max_val)))

col_fun <- colorRamp2(
  c(-max_abs, 0, max_abs),
  c("#7b3294", "#f7f7f7", "#008837")
)

# Render Heatmap
ht <- Heatmap(
  mat_weight,
  name = "Mean Edge Weight",
  col = col_fun,
  cluster_columns = TRUE,  # Preserve explicit condition order
  cluster_rows = TRUE,
  show_row_dend = TRUE,
  row_names_gp = gpar(fontsize = 9, fontface = "italic"),
  column_names_gp = gpar(fontsize = 9, fontface = "bold"),
  column_names_rot = 30,
  column_names_centered = FALSE,
  rect_gp = gpar(col = "grey90", lwd = 1),
  # cut tree rows into 5 clusters
  row_km = 5,
  # put more space between row clusters
  row_gap = unit(4, "mm"),
  # Draw prevalence percentage as overlay text inside each cell
  cell_fun = function(j, i, x, y, width, height, fill) {
    grid.text(
      sprintf("%.0f%%", mat_prev[i, j]), 
      x, y, 
      gp = gpar(fontsize = 7.5, col = ifelse(abs(mat_weight[i, j]) > (max_abs * 0.5), "white", "black"))
    )
  },
  heatmap_legend_param = list(
    title = "Mean SparCC weight",
    title_gp = gpar(fontsize = 9, fontface = "bold"),
    labels_gp = gpar(fontsize = 8),
    legend_height = unit(3.5, "cm")
  )
)

# Draw Final Combined Plot
draw(
  ht,
  row_title = "E. coli neighbor species (Top 50 by weight delta from Non-IBD)",
  row_title_gp = gpar(fontsize = 12, fontface = "bold"),
  heatmap_legend_side = "right"
)

# save plots
save_path <- "results/plots/analyses/ecoli"
if (!dir.exists(save_path)) {
  dir.create(save_path, recursive = TRUE)
}

#degree
ggsave(
  filename = file.path(save_path, "ecoli_degree_boxplots.pdf"),
  plot = degree_plot,
  width = 6,
  height = 6,
  dpi = 600
)

#betweenness
ggsave(
  filename = file.path(save_path, "ecoli_betweenness_boxplots.pdf"),
  plot = betweenness_plot,
  width = 6,
  height = 6,
  dpi = 600
)

# abundance
ggsave(
  filename = file.path(save_path, "ecoli_abundance_boxplots.pdf"),
  plot = abundance_plot,
  width = 6,
  height = 6,
  dpi = 600
)

# dysbiosis
ggsave(
  filename = file.path(save_path, "ecoli_dysbiosis_boxplots.pdf"),
  plot = dysbiosis_plot,
  width = 6,
  height = 6,
  dpi = 600
)

# similarity
ggsave(
  filename = file.path(save_path, "ecoli_cosine_similarity_boxplots.pdf"),
  plot = cossimil_plot,
  width = 6,
  height = 6,
  dpi = 600
)

# heatmap
pdf(
  file = file.path(save_path, "ecoli_edge_heatmap.pdf"),
  width = 8,
  height = 17
)
draw(
  ht,
  row_title = "E. coli neighbor species (Top 50 by weight delta from Non-IBD)",
  row_title_gp = gpar(fontsize = 12, fontface = "bold"),
  heatmap_legend_side = "right"
)
dev.off()

# Locate the project root (from any working directory inside the repo) and load shared paths
.root <- getwd(); while (!file.exists(file.path(.root, "code", "common", "paths.R")) && dirname(.root) != .root) .root <- dirname(.root)
source(file.path(.root, "code", "common", "paths.R"))

library(tidyverse)
library(igraph)
library(yardstick)
library(broom)
library(lsa)
library(ComplexHeatmap)
library(circlize)
library(pals)
library(patchwork)

# Load data
# ------------------------------------------------------------------------------
IBD_composite <- readRDS(file.path(DATA_ABUNDANCE, "IBD_studies_counts_filtered_composite.rds"))
HMP <- IBD_composite[["HMP_2019_ibdmdb"]]

# Load dysbiosis scores
HMP2_scores <- read_tsv(file.path(DATA_EXTERNAL, "HMP2_dysbiosis_scores.tsv"), col_names = FALSE) %>% 
  rename(`External ID` = X1, dysbiosis_score = X2, dysbiosis_state = X3)

HMP_timepoint_filtered <- HMP %>% 
  group_by(subject_id) %>% 
  filter(!duplicated(days_from_first_collection)) %>%
  filter(n_distinct(days_from_first_collection) >= 7) %>%
  arrange(subject_id, days_from_first_collection, id) %>%
  distinct(subject_id, days_from_first_collection, .keep_all = TRUE) %>%
  ungroup()

metadata_cols <- names(HMP_timepoint_filtered)[!startsWith(names(HMP_timepoint_filtered), "k__")]
taxa_cols <- names(HMP_timepoint_filtered)[startsWith(names(HMP_timepoint_filtered), "k__")]

HMP_metadata <- HMP_timepoint_filtered %>%
  select(all_of(metadata_cols)) %>% 
  mutate(condition = case_when(
    disease == "healthy" ~ "Non-IBD",
    disease == "UC" ~ "UC",
    disease == "CD" ~ "CD",
    TRUE ~ condition
  )) %>% 
  mutate(condition = factor(condition, levels = c("Non-IBD", "UC", "CD"))) %>% 
  left_join(HMP2_scores, by = c("id" = "External ID")) %>% 
  group_by(subject_id) %>% 
  mutate(mean_dysbiosis_score = mean(dysbiosis_score, na.rm = TRUE),
         median_dysbiosis_score = median(dysbiosis_score, na.rm = TRUE),
         pct_dysbiotic = mean(dysbiosis_state, na.rm = TRUE),
         ab_usage_proportion = mean(antibiotics_current_use == "yes") * 100) %>%
  ungroup() %>% 
  mutate(ab_usage_categorical = case_when(
    ab_usage_proportion == 0 ~ "0%",
    ab_usage_proportion > 0 & ab_usage_proportion <= 25 ~ "0%-25%",
    ab_usage_proportion > 25 & ab_usage_proportion <= 50 ~ "25%-50%",
    ab_usage_proportion > 50 & ab_usage_proportion <= 75 ~ "50%-75%",
    ab_usage_proportion > 75 & ab_usage_proportion <= 100 ~ "75%-100%"
  ))

# Get HMP Counts
HMP_counts <- HMP_timepoint_filtered %>%
  select(all_of(taxa_cols), id) %>%
  column_to_rownames("id")

# CLR transformation and aggregation at subject level
HMP_clr <- HMP_counts %>%
  vegan::decostand(method = "clr", pseudocount = 1)

HMP_clr_agg <- HMP_clr %>%
  as.data.frame() %>%
  rownames_to_column("id") %>%
  left_join(HMP_metadata %>%
              select(id, subject_id),
            by = "id") %>%
  select(-id) %>%
  group_by(subject_id) %>%
  summarise(across(everything(), mean), .groups = "drop") %>%
  column_to_rownames("subject_id")

PROCESSED_DATA_PATH <- file.path(DATA_PYG, "HMP_timepoints_self_w_functions_onehot", "gml", "0_3")

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

# RF FEATURES
# Load traditional model performances
baseline_results_path <- file.path(RES_TRAD_TABLES, "traditional_summary_metrics.csv")

valid_rf_runs <- read_csv(baseline_results_path) %>% 
  filter(val_type == "internal_hmp") %>% 
  select(where(~ !all(is.na(.)))) %>% 
  filter(model %in% c("RF")) %>% 
  filter(auprc >= 0.6) %>% 
  pull(run)

# make joint baseline metrics plot with graph-based models for comparison
trad_models_features_path <- file.path(RES_TRAD_TABLES, "traditional_selected_features.csv")
trad_models_features <- read_csv(trad_models_features_path) %>% 
  filter(model %in% c("RF")) %>%
  # Clean up taxon names
  mutate(feature = if_else(str_detect(feature, "s__"),
                           str_replace(feature, ".*s__", ""),
                           feature),
         feature = str_replace_all(feature, "_", " ")) %>% 
  # number of features selected per run
  group_by(run) %>%
  mutate(num_features = n()) %>%
  ungroup() %>% 
  filter(run %in% valid_rf_runs)

# Plot top 30 important features for RF
rf_features <- trad_models_features %>% 
  # scale between 0 and 1
  group_by(run) %>%
  mutate(importance_score = importance_score / max(importance_score)) %>%
  ungroup() %>%
  complete(run, feature, fill = list(importance_score = 0)) %>% # fill in missing features with 0 importance
  group_by(feature) %>%
  summarize(prevalence = sum(importance_score > 0) / n_distinct(run),
            mean_importance_score = mean(importance_score),
            .groups = "drop") %>%
  arrange(desc(mean_importance_score))

top_rf_features <- rf_features %>% slice_max(order_by = mean_importance_score, n = 20)

# NODES
# ------------------------------------------------------------------------------
parent_folder <- RES_INTERP_AGG
meta_cols <- c("graph_name", "logits", "true_label", "convergence_delta")

if (!file.exists(file.path(parent_folder, "all_seeds_attributions_raw.csv"))) {
  seed_dirs <- list.dirs(RES_INTERP_SEEDS, recursive = FALSE, full.names = TRUE)
  seed_dirs <- seed_dirs[grepl("seed_", basename(seed_dirs))][1:50]

  all_seeds_raw <- map_dfr(seed_dirs, function(s_dir) {
    seed_id <- basename(s_dir)
    files <- list.files(s_dir, pattern = "\\.csv$", full.names = TRUE)
    files_no_edges <- files[!grepl("_edges", files)]
    
    map_dfr(files_no_edges, function(fpath) {
      fname <- basename(fpath)
      gname <- str_remove(str_remove(fname, "^integrated_gradients_attributions_"), "\\.csv$")
      
      df <- read_csv(fpath, show_col_types = FALSE) %>% rename("node" = 1)
      prob_val <- 1 / (1 + exp(-df$logits[1]))
      
      df %>%
        select(-any_of(meta_cols)) %>%
        pivot_longer(cols = -node, names_to = "feature", values_to = "attribution") %>%
        group_by(node) %>%
        summarise(
          net_attr_seed   = sum(attribution, na.rm = TRUE),
          total_attr_seed = sum(abs(attribution, na.rm = TRUE)),
          .groups = "drop"
        ) %>%
        mutate(
          seed              = seed_id,
          graph_name        = gname,
          true_label        = factor(df$true_label[1], levels = c(1, 0)),
          prob              = prob_val,
          convergence_delta = df$convergence_delta[1]
        )
    })
  })
  write_csv(all_seeds_raw, file.path(parent_folder, "all_seeds_attributions_raw.csv"))
} else {
  all_seeds_raw <- read_csv(file.path(parent_folder, "all_seeds_attributions_raw.csv"), show_col_types = FALSE) %>% 
    mutate(true_label = factor(true_label, levels = c(1, 0)))
}

# Filter by seed performance
seed_auprc <- all_seeds_raw %>%
  distinct(seed, graph_name, true_label, prob) %>%
  group_by(seed) %>%
  pr_auc(truth = true_label, prob, event_level = "first") %>%
  rename(auprc = .estimate)

valid_seeds <- seed_auprc %>% filter(auprc >= 0.60) %>% pull(seed)

# Taxon name cleaner function
clean_node_name <- function(x) {
  x %>% 
    str_remove("^.*[s|g|f|o|c|p|k]__") %>% 
    str_replace_all("_", " ")
}

graph_node_summary <- all_seeds_raw %>%
  filter(seed %in% valid_seeds) %>%
  select(seed, graph_name, node, true_label, prob, net_attr_seed, total_attr_seed) %>%
  group_by(graph_name, node) %>%
  summarise(
    importance = mean(total_attr_seed, na.rm = TRUE),
    sd_importance = sd(total_attr_seed, na.rm = TRUE),
    net = mean(net_attr_seed, na.rm = TRUE),
    sd_net = sd(net_attr_seed, na.rm = TRUE),
    direction_consistency = abs(sum(net_attr_seed, na.rm = TRUE)) / sum(abs(net_attr_seed), na.rm = TRUE),
    n_seeds = sum(!is.na(net_attr_seed)),
    .groups = "drop"
  ) %>% 
  group_by(graph_name) %>%
  mutate(
    max_abs = max(importance, na.rm = TRUE),
    importance_norm = importance / max_abs,
    sd_importance_norm = sd_importance / max_abs,
    net_norm = net / max_abs,
    sd_net_norm = sd_net / max_abs
  ) %>%
  ungroup() %>%
  left_join(
    HMP_metadata %>% select(subject_id, condition) %>% distinct(subject_id, .keep_all = TRUE), 
    by = c("graph_name" = "subject_id")
  ) %>%
  mutate(
    condition_broad = case_when(
      condition == "Non-IBD" ~ "Non-IBD",
      condition %in% c("UC", "CD") ~ "IBD",
      TRUE ~ "Other"
    ),
    node_clean = clean_node_name(node)
  )

# Matrix & annotation preparation
# ------------------------------------------------------------------------------
total_study_subjects <- n_distinct(graph_node_summary$graph_name)

node_prevalence_df <- graph_node_summary %>%
  group_by(node_clean) %>%
  summarise(
    occurrence = n_distinct(graph_name),
    prevalence = occurrence / total_study_subjects,
    mean_importance = mean(importance_norm, na.rm = TRUE),
    prevalence_weighted_importance = prevalence * mean_importance,
    mean_direction_consistency = mean(direction_consistency, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  left_join(
    top_rf_features %>% select(feature, mean_importance_score),
    by = c("node_clean" = "feature")
  ) %>%
  arrange(desc(mean_importance_score))

# Explicitly retain nodes from RF top features
rf_feat <- node_prevalence_df %>% 
  filter(node_clean %in% top_rf_features$feature) %>%
  pull(node_clean)

gnn_feat <- node_prevalence_df %>% 
  slice_max(order_by = prevalence_weighted_importance, n = 20) %>%
  pull(node_clean)

top_nodes <- unique(c(rf_feat, gnn_feat))

mat_weighted <- graph_node_summary %>%
  filter(node_clean %in% top_nodes) %>%
  mutate(node_clean = factor(node_clean, levels = top_nodes)) %>%
  select(node_clean, graph_name, net_norm) %>%
  pivot_wider(names_from = graph_name, values_from = net_norm, values_fill = 0) %>%
  arrange(node_clean) %>%
  column_to_rownames("node_clean") %>%
  as.matrix()

subject_conditions <- graph_node_summary %>% distinct(graph_name, condition_broad)
total_subjects <- subject_conditions %>% count(condition_broad) %>% deframe()

condition_prevalence_indep <- graph_node_summary %>%
  filter(node_clean %in% top_nodes) %>%
  group_by(node_clean, condition_broad) %>%
  summarise(count = n_distinct(graph_name), .groups = "drop") %>%
  filter(condition_broad %in% c("Non-IBD", "IBD")) %>%
  mutate(rate = count / total_subjects[condition_broad]) %>%
  pivot_wider(id_cols = node_clean, names_from = condition_broad, values_from = rate, values_fill = 0) %>%
  slice(match(rownames(mat_weighted), node_clean)) %>%
  column_to_rownames("node_clean") %>%
  select(`Non-IBD`, `IBD`) %>%
  as.matrix()

# Prepare full distributions for boxplots
raw_importance_df <- graph_node_summary %>%
  filter(node_clean %in% top_nodes) %>%
  select(node_clean, graph_name, importance_norm) %>%
  pivot_wider(names_from = graph_name, values_from = importance_norm, values_fill = 0) %>%
  slice(match(rownames(mat_weighted), node_clean)) %>%
  column_to_rownames("node_clean") %>%
  as.matrix()

raw_stability_df <- graph_node_summary %>%
  filter(node_clean %in% top_nodes) %>%
  select(node_clean, graph_name, direction_consistency) %>%
  pivot_wider(names_from = graph_name, values_from = direction_consistency, values_fill = 0) %>%
  slice(match(rownames(mat_weighted), node_clean)) %>%
  column_to_rownames("node_clean") %>%
  as.matrix()

# Metadata aligned to matrix columns
graph_meta <- HMP_metadata %>%
  filter(subject_id %in% colnames(mat_weighted)) %>%
  distinct(subject_id, .keep_all = TRUE) %>%
  mutate(
    condition_broad = case_when(
      disease == "healthy" | condition == "Non-IBD" ~ "Non-IBD",
      disease %in% c("UC", "CD") | condition %in% c("UC", "CD") ~ "IBD",
      TRUE ~ "Other"
    ),
    ab_usage_categorical = if_else(is.na(ab_usage_categorical), "Unknown", as.character(ab_usage_categorical)),
    age_category         = if_else(is.na(age_category), "Unknown", str_to_title(age_category)),
    gender               = if_else(is.na(gender), "Unknown", str_to_title(gender))
  ) %>%
  slice(match(colnames(mat_weighted), subject_id))

# Compute metrics: Timepoints, Shannon & Median CLR
# ------------------------------------------------------------------------------
clr_long <- HMP_clr_agg %>%
  rownames_to_column("subject_id") %>%
  left_join(
    graph_meta %>% select(subject_id, condition_broad),
    by = "subject_id"
  ) %>%
  filter(condition_broad %in% c("Non-IBD", "IBD")) %>%
  pivot_longer(
    cols = starts_with("k__"),
    names_to = "node",
    values_to = "clr"
  ) %>%
  mutate(node_clean = clean_node_name(node)) %>%
  filter(node_clean %in% top_nodes)

median_clr_mat <- clr_long %>%
  group_by(node_clean, condition_broad) %>%
  summarise(median_clr = median(clr, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = condition_broad, values_from = median_clr, values_fill = 0) %>%
  slice(match(rownames(mat_weighted), node_clean)) %>%
  column_to_rownames("node_clean") %>%
  select(`Non-IBD`, `IBD`) %>%
  as.matrix()

# Compute Condition-specific Median Normalized Degree
# ------------------------------------------------------------------------------
degree_df <- map_dfr(names(igraph_list), function(gname) {
  g <- igraph_list[[gname]]
  if (is.null(g)) return(NULL)
  
  v_names <- V(g)$name
  if (is.null(v_names)) v_names <- V(g)$label
  
  data.frame(
    graph_name = gname,
    node = v_names,
    degree_norm = igraph::degree(g, v = V(g), mode = "all", normalized = TRUE),
    stringsAsFactors = FALSE
  )
}) %>%
  mutate(node_clean = clean_node_name(node)) %>%
  left_join(
    graph_meta %>% select(subject_id, condition_broad),
    by = c("graph_name" = "subject_id")
  ) %>%
  filter(condition_broad %in% c("Non-IBD", "IBD"))

median_norm_degree_mat <- degree_df %>%
  filter(node_clean %in% top_nodes) %>%
  group_by(node_clean, condition_broad) %>%
  summarise(med_deg = median(degree_norm, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = condition_broad, values_from = med_deg, values_fill = 0) %>%
  slice(match(rownames(mat_weighted), node_clean)) %>%
  column_to_rownames("node_clean") %>%
  select(`Non-IBD`, `IBD`) %>%
  as.matrix()

# Compute Condition-specific Median UNWEIGHTED Betweenness Centrality
# ------------------------------------------------------------------------------
betweenness_df <- map_dfr(names(igraph_list), function(gname) {
  g <- igraph_list[[gname]]
  if (is.null(g)) return(NULL)
  
  v_names <- V(g)$name
  if (is.null(v_names)) v_names <- V(g)$label
  
  data.frame(
    graph_name = gname,
    node = v_names,
    betweenness_norm = igraph::betweenness(
      g, 
      v = V(g), 
      directed = FALSE, 
      weights = NA, 
      normalized = TRUE
    ),
    stringsAsFactors = FALSE
  )
}) %>%
  mutate(node_clean = clean_node_name(node)) %>%
  left_join(
    graph_meta %>% select(subject_id, condition_broad),
    by = c("graph_name" = "subject_id")
  ) %>%
  filter(condition_broad %in% c("Non-IBD", "IBD"))

median_betweenness_mat <- betweenness_df %>%
  filter(node_clean %in% top_nodes) %>%
  group_by(node_clean, condition_broad) %>%
  summarise(med_bet = median(betweenness_norm, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = condition_broad, values_from = med_bet, values_fill = 0) %>%
  slice(match(rownames(mat_weighted), node_clean)) %>%
  column_to_rownames("node_clean") %>%
  select(`Non-IBD`, `IBD`) %>%
  as.matrix()

# Colors
condition_broad_colors <- c("Non-IBD" = "#0F4C81", "IBD" = "#D90429", "Other" = "grey80")

# Align RF Data to top_nodes
# ------------------------------------------------------------------------------
rf_importance_df <- rf_features %>%
  filter(feature %in% top_nodes) %>%
  slice(match(top_nodes, feature))

rf_imp_vec <- rf_importance_df$mean_importance_score
names(rf_imp_vec) <- rf_importance_df$feature

rf_prev_vec <- rf_importance_df$prevalence
names(rf_prev_vec) <- rf_importance_df$feature

# Left Annotation Construction (Prevalence -> CLR -> Degree -> Betweenness)
# ------------------------------------------------------------------------------
left_anno <- rowAnnotation(
  `Prevalence rate\n(per condition)` = anno_barplot(
    condition_prevalence_indep,
    beside = TRUE,
    attach = TRUE,
    gp     = gpar(fill = unname(condition_broad_colors[c("Non-IBD", "IBD")]), col = NA),
    ylim   = c(0, 1),
    width  = unit(2.2, "cm")
  ),
  `Median CLR` = anno_points(
    median_clr_mat,
    gp    = gpar(col = unname(condition_broad_colors[c("Non-IBD", "IBD")])),
    pch   = 16,
    size  = unit(2.5, "mm"),
    width = unit(2.2, "cm")
  ),
  `Norm. degree` = anno_points(
    median_norm_degree_mat,
    gp    = gpar(col = unname(condition_broad_colors[c("Non-IBD", "IBD")])),
    pch   = 16,
    size  = unit(2.5, "mm"),
    width = unit(2.2, "cm")
  ),
  `Norm. betweenness` = anno_points(
    median_betweenness_mat,
    gp    = gpar(col = unname(condition_broad_colors[c("Non-IBD", "IBD")])),
    pch   = 16,
    size  = unit(2.5, "mm"),
    width = unit(2.2, "cm")
  ),
  annotation_name_side = "top",
  annotation_name_gp   = gpar(fontsize = 8.5, fontface = "bold")
)

# Right Annotation Construction (RF Importance -> RF Prevalence -> Importance -> Stability)
# ------------------------------------------------------------------------------
right_anno <- rowAnnotation(
  `RF Mean Importance` = anno_barplot(
    rf_imp_vec,
    gp     = gpar(fill = "#2B5C8F", col = NA),
    width  = unit(3.0, "cm"),
    border = FALSE
  ),
  `Absolute importance` = anno_boxplot(
    raw_importance_df,
    gp        = gpar(fill = "#494949ff", col = "black"),
    width     = unit(2.2, "cm"),
    box_width = 0.6,
    outline   = TRUE,
    pch       = 16,
    size      = unit(0.5, "mm")
  ),
  `Direction stability` = anno_boxplot(
    raw_stability_df,
    gp        = gpar(fill = "#2A9D8F", col = "black"),
    ylim      = c(0, 1),
    width     = unit(2.2, "cm"),
    box_width = 0.6,
    outline   = TRUE,
    pch       = 16,
    size      = unit(0.5, "mm")
  ),
  annotation_name_side = "top",
  annotation_name_gp   = gpar(fontsize = 8.5, fontface = "bold")
)

# Render Plot (using a zero-width dummy base heatmap)
# ------------------------------------------------------------------------------
dummy_mat <- matrix(0, nrow = length(top_nodes), ncol = 1)
rownames(dummy_mat) <- top_nodes

# Determine fontfaces: bold italic for GNN top features, regular italic otherwise
gnn_top_features <- gnn_feat
row_fontfaces <- if_else(rownames(dummy_mat) %in% gnn_top_features, "bold.italic", "italic")

ht <- Heatmap(
  dummy_mat,
  name                = "base",
  left_annotation     = left_anno,
  right_annotation    = right_anno,
  cluster_rows        = FALSE,  # Retains exact top_nodes ranking
  cluster_columns     = FALSE,
  show_column_names   = FALSE,
  show_heatmap_legend = FALSE,
  rect_gp             = gpar(type = "none"),
  width               = unit(0, "mm"),
  row_names_gp        = gpar(fontsize = 10, fontface = row_fontfaces),
  height              = unit(18, "cm")
)

lgd_cond_points <- Legend(
  labels    = c("Non-IBD", "IBD"),
  title     = "Condition Metrics",
  type      = "points",
  pch       = 16,
  legend_gp = gpar(col = unname(condition_broad_colors[c("Non-IBD", "IBD")]))
)

draw(
  ht, 
  annotation_legend_list = list(lgd_cond_points),
  ht_gap = unit(3, "mm")
)

# Statistical tests for RF Features Only (Prevalence, CLR, Degree, Betweenness)
# ------------------------------------------------------------------------------

# Define RF features target vector
rf_nodes <- rf_feat  # Alternatively: unique(top_rf_features$feature)

# Prepare Subject-Level Mapping
subject_cond_map <- graph_meta %>%
  select(subject_id, condition_broad) %>%
  filter(condition_broad %in% c("Non-IBD", "IBD")) %>%
  mutate(condition_broad = factor(condition_broad, levels = c("Non-IBD", "IBD")))

all_subjects <- subject_cond_map$subject_id

# Extract total subjects per group
n_tot_ibd <- sum(subject_cond_map$condition_broad == "IBD")
n_tot_non <- sum(subject_cond_map$condition_broad == "Non-IBD")

# 1) Fisher's Exact Test for Prevalence Differences
full_presence_df <- expand_grid(
  node_clean = rf_nodes,
  graph_name = all_subjects
) %>%
  left_join(subject_cond_map, by = c("graph_name" = "subject_id")) %>%
  left_join(
    graph_node_summary %>% 
      filter(node_clean %in% rf_nodes) %>% 
      distinct(node_clean, graph_name) %>% 
      mutate(present = 1),
    by = c("node_clean", "graph_name")
  ) %>%
  mutate(present = replace_na(present, 0))

mwu_prevalence <- full_presence_df %>%
  group_by(node_clean) %>%
  summarise(
    n_present_IBD     = sum(present == 1 & condition_broad == "IBD"),
    n_present_Non_IBD = sum(present == 1 & condition_broad == "Non-IBD"),
    N_tot_IBD         = n_tot_ibd,
    N_tot_Non_IBD     = n_tot_non,
    
    # Construct 2x2 matrix directly inside fisher.test
    fisher_res = list(fisher.test(
      matrix(c(
        n_present_IBD, N_tot_IBD - n_present_IBD,
        n_present_Non_IBD, N_tot_Non_IBD - n_present_Non_IBD
      ), nrow = 2, byrow = TRUE)
    )),
    
    # Extract p-value and uncorrected Odds Ratio (>1 = enriched in IBD)
    pval_prevalence = fisher_res[[1]]$p.value,
    OR_prevalence   = as.numeric(fisher_res[[1]]$estimate),
    .groups = "drop"
  ) %>%
  select(
    node_clean, N_tot_IBD, N_tot_Non_IBD, n_present_IBD, n_present_Non_IBD, 
    OR_prevalence, pval_prevalence
  ) %>%
  mutate(padj_prevalence = p.adjust(pval_prevalence, method = "BH"))

# 2) CLR Abundance Differences
full_clr_df <- clr_long %>%
  filter(node_clean %in% rf_nodes) %>%
  inner_join(subject_cond_map, by = c("subject_id", "condition_broad"))

mwu_clr <- full_clr_df %>%
  group_by(node_clean) %>%
  summarise(
    test = list(wilcox.test(clr ~ condition_broad, data = pick(everything()), exact = FALSE)),
    .groups = "drop"
  ) %>%
  mutate(tidy_res = map(test, broom::tidy)) %>%
  unnest(tidy_res) %>%
  select(node_clean, statistic, p.value) %>%
  rename(W_clr = statistic, pval_clr = p.value) %>%
  mutate(padj_clr = p.adjust(pval_clr, method = "BH"))

# 3) Normalized Degree Differences
full_degree_df <- expand_grid(
  node_clean = rf_nodes,
  graph_name = all_subjects
) %>%
  left_join(subject_cond_map, by = c("graph_name" = "subject_id")) %>%
  left_join(
    degree_df %>% filter(node_clean %in% rf_nodes) %>% select(node_clean, graph_name, degree_norm),
    by = c("node_clean", "graph_name")
  ) %>%
  mutate(degree_norm = replace_na(degree_norm, 0))

mwu_degree <- full_degree_df %>%
  group_by(node_clean) %>%
  summarise(
    test = list(wilcox.test(degree_norm ~ condition_broad, data = pick(everything()), exact = FALSE)),
    .groups = "drop"
  ) %>%
  mutate(tidy_res = map(test, broom::tidy)) %>%
  unnest(tidy_res) %>%
  select(node_clean, statistic, p.value) %>%
  rename(W_degree = statistic, pval_degree = p.value) %>%
  mutate(padj_degree = p.adjust(pval_degree, method = "BH"))

# 4) Normalized Betweenness Differences
full_betweenness_df <- expand_grid(
  node_clean = rf_nodes,
  graph_name = all_subjects
) %>%
  left_join(subject_cond_map, by = c("graph_name" = "subject_id")) %>%
  left_join(
    betweenness_df %>% filter(node_clean %in% rf_nodes) %>% select(node_clean, graph_name, betweenness_norm),
    by = c("node_clean", "graph_name")
  ) %>%
  mutate(betweenness_norm = replace_na(betweenness_norm, 0))

mwu_betweenness <- full_betweenness_df %>%
  group_by(node_clean) %>%
  summarise(
    test = list(wilcox.test(betweenness_norm ~ condition_broad, data = pick(everything()), exact = FALSE)),
    .groups = "drop"
  ) %>%
  mutate(tidy_res = map(test, broom::tidy)) %>%
  unnest(tidy_res) %>%
  select(node_clean, statistic, p.value) %>%
  rename(W_betweenness = statistic, pval_betweenness = p.value) %>%
  mutate(padj_betweenness = p.adjust(pval_betweenness, method = "BH"))

# Combine and print final summary table
mwu_rf_summary_table <- tibble(node_clean = rf_nodes) %>%
  left_join(mwu_prevalence, by = "node_clean") %>%
  left_join(mwu_clr, by = "node_clean") %>%
  left_join(mwu_degree, by = "node_clean") %>%
  left_join(mwu_betweenness, by = "node_clean") %>%
  mutate(node_clean = factor(node_clean, levels = rf_nodes)) %>%
  arrange(node_clean)

# Print Final Result Table in R Console
print(mwu_rf_summary_table, n = length(rf_nodes))

# ------------------------------------------------------------------------------
# CLR Boxplots for Top Nodes (Bolding GNN Features)
# ------------------------------------------------------------------------------

# Filter long CLR dataset for top features and ensure factor ordering
clr_plot_df <- clr_long %>%
  filter(node_clean %in% top_nodes) %>%
  mutate(
    node_clean = factor(node_clean, levels = rev(top_nodes)), # Reverse for top-to-bottom layout
    condition_broad = factor(condition_broad, levels = c("Non-IBD", "IBD"))
  )

# Determine axis text face dynamically: bold.italic for GNN features, italic otherwise
# (Order must match rev(top_nodes) corresponding to bottom-to-top axis rendering)
axis_fontfaces <- if_else(rev(top_nodes) %in% gnn_feat, "bold.italic", "italic")

# Build Boxplot
p_clr_boxplots <- ggplot(clr_plot_df, aes(x = clr, y = node_clean, fill = condition_broad)) +
  geom_boxplot(
# do not show outliers
    outlier.shape = NA,
    alpha = 0.85,
    position = position_dodge(width = 0.75),
    width = 0.6
  ) +
  scale_fill_manual(values = condition_broad_colors) +
  labs(
    x = "Centered Log-Ratio (CLR) Abundance",
    y = NULL,
    fill = "Condition"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.y = element_text(face = axis_fontfaces, color = "black", size = 10),
    axis.title.x = element_text(face = "bold", margin = margin(t = 8)),
    legend.title = element_text(face = "bold"),
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank()
  )

# Display Plot
print(p_clr_boxplots)

# Save plots
save_path = FIG_RF_FEATURES
if (!dir.exists(save_path)) {
  dir.create(save_path, recursive = TRUE)
}

ggsave(
  filename = file.path(save_path, "clr_boxplots_top_RF.pdf"),
  plot = p_clr_boxplots,
  width = 5,
  height = 7,
  units = "in"
)
pdf(file = file.path(save_path, "heatmap_top_RF.pdf"), width = 14, height = 9, useDingbats = FALSE)
draw(
  ht, 
  annotation_legend_list = list(lgd_cond_points),
  ht_gap = unit(3, "mm")
)
dev.off()


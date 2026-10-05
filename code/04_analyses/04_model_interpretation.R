library(tidyverse)
library(ggraph)
library(ggrepel)
library(tidygraph)
library(igraph)
library(plotROC)
library(yardstick)
library(broom)
library(lsa)
library(ComplexHeatmap)
library(circlize)
library(pals)
library(patchwork)

# Load data
# ------------------------------------------------------------------------------
IBD_composite <- readRDS("data/processed/IBD_CuratedMetagenomicData/IBD_studies_counts_filtered_composite.rds")
HMP <- IBD_composite[["HMP_2019_ibdmdb"]]

# Load dysbiosis scores
HMP2_scores <- read_tsv("data/external/HMP2_dysbiosis_scores.tsv", col_names = FALSE) %>% 
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

PROCESSED_DATA_PATH <- file.path("data", "processed",
  "pyg_datasets", "HMP_timepoints_self_w_functions_onehot", "gml", "0_3")

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

# NODES
# ------------------------------------------------------------------------------
parent_folder <- "results/machine_learning/graph_based_models/model_interpretation/HMP_timepoints_self_t_0_3/interpretation_10_seeds"
meta_cols <- c("graph_name", "logits", "true_label", "convergence_delta")

if (!file.exists(file.path(parent_folder, "all_seeds_attributions_raw.csv"))) {
  seed_dirs <- list.dirs(parent_folder, recursive = FALSE, full.names = TRUE)
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
          total_attr_seed = sum(abs(attribution), na.rm = TRUE),
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
  arrange(desc(prevalence_weighted_importance))

# Explicitly retain ordered top 20 nodes
top_nodes <- head(node_prevalence_df$node_clean, 20)

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
      condition == "Non-IBD" ~ "Non-IBD",
      condition %in% c("UC", "CD") ~ "IBD",
      TRUE ~ "Other"
    ),
    ab_usage_categorical = if_else(is.na(ab_usage_categorical), "Unknown", as.character(ab_usage_categorical)),
    age_category         = if_else(is.na(age_category), "Unknown", str_to_title(age_category)),
    gender               = if_else(is.na(gender), "Unknown", str_to_title(gender))
  ) %>%
  slice(match(colnames(mat_weighted), subject_id))

# Compute metrics: Timepoints, Shannon & Median CLR
# ------------------------------------------------------------------------------
timepoint_df <- HMP_timepoint_filtered %>%
  group_by(subject_id) %>%
  summarise(n_timepoints = n_distinct(days_from_first_collection), .groups = "drop")

timepoint_vec <- timepoint_df$n_timepoints[match(colnames(mat_weighted), timepoint_df$subject_id)]

shannon_df <- HMP_counts %>%
  mutate(shannon = vegan::diversity(select(., -all_of(contains("id"))), index = "shannon")) %>%
  rownames_to_column("id") %>%
  left_join(HMP_metadata %>% select(id, subject_id), by = "id") %>%
  group_by(subject_id) %>%
  summarise(mean_shannon = mean(shannon, na.rm = TRUE), .groups = "drop")

shannon_vec <- shannon_df$mean_shannon[match(colnames(mat_weighted), shannon_df$subject_id)]

# Condition-specific Median Aggregated CLR Abundance
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
subtype_colors <- c("UC" = "#F8961E", "CD" = "#93051c", "Non-IBD" = "#0F4C81", "Other" = "grey80")
gender_colors          <- c("Male" = "#2A9D8F", "Female" = "#B565A7", "Unknown" = "grey80")
age_colors             <- c("Adult" = "#7209B7", "Child" = "#4EA8DE", "Schoolage" = "#9EF01A", "Senior" = "#F15BB5", "Unknown" = "grey80")
ab_colors              <- c("0%" = "#3A86C8", "0%-25%" = "#90BE6D", "25%-50%" = "#F9C74F", "50%-75%" = "#F8961E", "75%-100%" = "#C93B32", "Unknown" = "grey80")

# Top Annotation Construction
# ------------------------------------------------------------------------------
top_anno <- HeatmapAnnotation(
  Condition      = graph_meta$condition_broad,
  Subtype        = graph_meta$condition,
  Antibiotics    = graph_meta$ab_usage_categorical,
  Sex            = graph_meta$gender,
  `Age category` = graph_meta$age_category,
  `Timepoints`   = anno_barplot(
    timepoint_vec,
    gp        = gpar(fill = "#2B9348", col = NA),
    height    = unit(1.5, "cm"),
    add_lines = TRUE,
    baseline  = 0
  ),
  `Shannon` = anno_barplot(
    shannon_vec,
    gp     = gpar(fill = "#5A189A", col = NA),
    height = unit(1.5, "cm")
  ),
  col = list(
    Condition   = condition_broad_colors,
    Subtype     = subtype_colors,
    Antibiotics = ab_colors,
    Sex         = gender_colors,
    `Age category` = age_colors
  ),
  annotation_name_side = "left",
  annotation_name_gp   = gpar(fontsize = 9, fontface = "bold"),
  gap                  = unit(c(1, 1, 1, 1, 2, 2), "mm")
)

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

# Right Annotation Construction (Importance -> Stability)
# ------------------------------------------------------------------------------
right_anno <- rowAnnotation(
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

# Render Heatmap
# ------------------------------------------------------------------------------
color_cap <- quantile(abs(mat_weighted[mat_weighted != 0]), 0.95, na.rm = TRUE)
col_fun <- circlize::colorRamp2(c(-color_cap, 0, color_cap), c("#D90429", "white", "#0F4C81"))

ht <- Heatmap(
  mat_weighted,
  name              = "Signed\nimportance",
  col               = col_fun,
  top_annotation    = top_anno,
  left_annotation   = left_anno,
  right_annotation  = right_anno,
  cluster_rows      = FALSE,  # Retains exact top_nodes ranking
  cluster_columns   = TRUE,
  show_column_names = FALSE,
  row_names_gp      = gpar(fontsize = 10, fontface = "italic"),
  rect_gp           = gpar(col = "white", lwd = 0.5),
  height            = unit(18, "cm")
)

lgd_cond_points <- Legend(
  labels    = c("Non-IBD", "IBD"),
  title     = "Condition Metrics",
  type      = "points",
  pch       = 16,
  legend_gp = gpar(col = unname(condition_broad_colors[c("Non-IBD", "IBD")]))
)

draw(ht, merge_legend = TRUE, annotation_legend_list = list(lgd_cond_points))

# Statistical tests for Top 20 Nodes (Prevalence, CLR, Degree, Betweenness)
# ------------------------------------------------------------------------------

# Prepare Subject-Level Mapping
subject_cond_map <- graph_meta %>%
  select(subject_id, condition_broad) %>%
  filter(condition_broad %in% c("Non-IBD", "IBD")) %>%
  mutate(condition_broad = factor(condition_broad, levels = c("Non-IBD", "IBD")))

all_subjects <- subject_cond_map$subject_id

# Extract total subjects per group
n_tot_ibd <- sum(subject_cond_map$condition_broad == "IBD")
n_tot_non <- sum(subject_cond_map$condition_broad == "Non-IBD")

# 1) Fisher's Exact Test for Prevalence Differences (Uncorrected OR + Sample Sizes)
full_presence_df <- expand_grid(
  node_clean = top_nodes,
  graph_name = all_subjects
) %>%
  left_join(subject_cond_map, by = c("graph_name" = "subject_id")) %>%
  left_join(
    graph_node_summary %>% 
      filter(node_clean %in% top_nodes) %>% 
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
  filter(node_clean %in% top_nodes) %>%
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
  node_clean = top_nodes,
  graph_name = all_subjects
) %>%
  left_join(subject_cond_map, by = c("graph_name" = "subject_id")) %>%
  left_join(
    degree_df %>% filter(node_clean %in% top_nodes) %>% select(node_clean, graph_name, degree_norm),
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
  node_clean = top_nodes,
  graph_name = all_subjects
) %>%
  left_join(subject_cond_map, by = c("graph_name" = "subject_id")) %>%
  left_join(
    betweenness_df %>% filter(node_clean %in% top_nodes) %>% select(node_clean, graph_name, betweenness_norm),
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
mwu_top20_summary_table <- tibble(node_clean = top_nodes) %>%
  left_join(mwu_prevalence, by = "node_clean") %>%
  left_join(mwu_clr, by = "node_clean") %>%
  left_join(mwu_degree, by = "node_clean") %>%
  left_join(mwu_betweenness, by = "node_clean") %>%
  mutate(node_clean = factor(node_clean, levels = top_nodes)) %>%
  arrange(node_clean)

# Print Final Result Table in R Console
print(mwu_top20_summary_table, n = 20)

# Correlations between centrality, abundance and importance (Colored by Order)
# ------------------------------------------------------------------------------

# 1. Compute Betweenness Centrality for All Nodes
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
  mutate(node_clean = clean_node_name(node))

median_betweenness_df <- betweenness_df %>%
  group_by(node_clean) %>%
  summarise(
    median_norm_betweenness = median(betweenness_norm, na.rm = TRUE),
    .groups = "drop"
  )

# 2. Calculate mean CLR abundance for ALL taxa
clr_all_taxa_long <- HMP_clr_agg %>%
  rownames_to_column("subject_id") %>%
  pivot_longer(
    cols = starts_with("k__"),
    names_to = "node",
    values_to = "clr"
  ) %>%
  mutate(node_clean = clean_node_name(node)) %>%
  group_by(node_clean) %>%
  summarise(
    median_clr_all = median(clr, na.rm = TRUE),
    .groups = "drop"
  )

# 3. Extract taxonomic Order mapping for all nodes
node_order_map <- graph_node_summary %>%
  distinct(node_clean, node) %>%
  separate(node, into = c("kingdom", "phylum", "class", "order", "family", "genus", "species"), sep = "\\|") %>%
  mutate(order = str_remove(order, "^o__"),
         order = if_else(is.na(order) | order == "", "Unclassified", order)) %>%
  select(node_clean, order)

# 4. Merge metrics, flag Top 20, and join Order metadata
scatter_master_df <- graph_node_summary %>%
  left_join(
    degree_df %>%
      group_by(node_clean) %>%
      summarise(
        median_norm_degree = median(degree_norm, na.rm = TRUE),
        mean_norm_degree   = mean(degree_norm, na.rm = TRUE),
        .groups = "drop"
      ),
    by = "node_clean"
  ) %>%
  left_join(median_betweenness_df, by = "node_clean") %>%
  left_join(clr_all_taxa_long, by = "node_clean") %>%
  group_by(node_clean) %>%
  summarise(
    occurrence                   = n_distinct(graph_name),
    prevalence                   = occurrence / total_study_subjects,
    mean_abs_importance           = mean(importance_norm, na.rm = TRUE),
    weighted_mean_abs_importance = prevalence * mean_abs_importance,
    median_norm_degree           = coalesce(max(median_norm_degree, na.rm = TRUE), 0),
    mean_norm_degree             = coalesce(max(mean_norm_degree, na.rm = TRUE), 0),
    median_norm_betweenness      = coalesce(max(median_norm_betweenness, na.rm = TRUE), 0),
    median_clr_all               = mean(median_clr_all, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(!is.na(median_clr_all)) %>%
  mutate(is_top20 = node_clean %in% top_nodes) %>%
  left_join(node_order_map, by = "node_clean")

# 5. Explicit color palette mapping based on reference metagraph
image_order_colors <- c(
  "Enterobacterales"        = "#FF0000",  # Bright Red
  "Enterobacterioles"       = "#FF0000",
  "Bacteroidales"           = "#1E88E5",  # Primary Blue
  "Desulfovibrionales"      = "#00A896",  # Distinct Teal / Cyan
  "Clostridiales"           = "#FF8C00",  # Orange
  "Burkholderiales"         = "#388E3C",  # Green
  "Bifidobacteriales"       = "#7B1FA2",  # Purple
  "Lactobacillales"         = "#E91E63",  # Magenta / Pink
  "Erysipelotrichales"      = "#F4511E",  # Red-Orange
  "Verrucomicrobiales"      = "#00ACC1",  # Soft Cyan
  "Coriobacteriales"        = "#8D6E63",  # Brown
  "Proteobacteria_unclassified" = "#616161",# Dark Grey
  "Unclassified"            = "#9E9E9E"   # Grey
)

# Assign palette values to present orders, adding fallback colors for unlisted orders
unique_orders <- unique(scatter_master_df$order[scatter_master_df$is_top20])
unmapped_orders <- setdiff(unique_orders, names(image_order_colors))

if (length(unmapped_orders) > 0) {
  fallback_cols <- setNames(pals::cols25(length(unmapped_orders)), unmapped_orders)
  final_order_colors <- c(image_order_colors, fallback_cols)
} else {
  final_order_colors <- image_order_colors
}

# 6. Helper function to generate scatter plots with overlap-free repelled labels
make_custom_scatter <- function(data, x_var, y_var, x_label, y_label, show_legend = FALSE) {
  x_sym <- sym(x_var)
  y_sym <- sym(y_var)
  
  # Compute Spearman Correlation
  cor_res <- cor.test(data[[x_var]], data[[y_var]], method = "spearman", exact = FALSE)
  rho_val <- round(cor_res$estimate, 2)
  p_val   <- format.pval(cor_res$p.value, digits = 2)
  
  # Position annotations
  x_vals <- data[[x_var]][!is.na(data[[x_var]])]
  y_vals <- data[[y_var]][!is.na(data[[y_var]])]
  
  x_min <- min(x_vals); x_max <- max(x_vals)
  y_min <- min(y_vals); y_max <- max(y_vals)
  
  x_pos <- x_min + 0.03 * (x_max - x_min)
  y_pos <- y_max - 0.03 * (y_max - y_min)
  
  ggplot(data, aes(x = !!x_sym, y = !!y_sym)) +
    # Linear trend line
    geom_smooth(method = "lm", linetype = "dashed", color = "gray40", se = FALSE) +
    # Background nodes in neutral grey
    geom_point(
      data = filter(data, !is_top20),
      color = "grey80",
      alpha = 0.5,
      size = 1.2
    ) +
    # Highlighted top 20 nodes with reference order colors
    geom_point(
      data = filter(data, is_top20),
      aes(color = order),
      alpha = 0.95,
      size = 2.8
    ) +
    # Overlap-proof labels for top 20 nodes
    geom_text_repel(
      data = filter(data, is_top20),
      aes(label = node_clean),
      size = 2.8,
      fontface = "italic",
      color = "black",
      bg.color = "white",      
      bg.r = 0.12,
      max.overlaps = Inf,      
      box.padding = 0.5,        
      point.padding = 0.3,     
      force = 4,                
      force_pull = 1,           
      min.segment.length = 0,   
      segment.color = "gray50",
      segment.alpha = 0.7
    ) +
    annotate(
      "text", x = x_pos, y = y_pos,
      label = paste0("Spearman r = ", rho_val, "\n(p = ", p_val, ")"),
      hjust = 0, vjust = 1, size = 3.5, fontface = "bold"
    ) +
    scale_color_manual(values = final_order_colors, name = "Order", breaks = unique_orders) +
    scale_x_continuous(expand = expansion(mult = c(0.12, 0.12))) +
    scale_y_continuous(expand = expansion(mult = c(0.12, 0.12))) +
    labs(x = x_label, y = y_label) +
    theme_classic(base_size = 11) +
    theme(
      axis.title = element_text(face = "bold"),
      legend.position = if (show_legend) "right" else "none"
    )
}

# 7. Build Scatter Plots
p_imp_deg <- make_custom_scatter(
  scatter_master_df,
  x_var = "median_norm_degree",
  y_var = "weighted_mean_abs_importance",
  x_label = "Median normalized degree",
  y_label = "Prevalence-weighted mean abs. importance"
)

p_imp_bet <- make_custom_scatter(
  scatter_master_df,
  x_var = "median_norm_betweenness",
  y_var = "weighted_mean_abs_importance",
  x_label = "Median normalized betweenness",
  y_label = "Prevalence-weighted mean abs. importance"
)

p_deg_bet <- make_custom_scatter(
  scatter_master_df,
  x_var = "median_norm_degree",
  y_var = "median_norm_betweenness",
  x_label = "Median normalized degree",
  y_label = "Median normalized betweenness"
)

p_clr_imp <- make_custom_scatter(
  scatter_master_df,
  x_var = "median_clr_all",
  y_var = "weighted_mean_abs_importance",
  x_label = "Median CLR-transformed abundance",
  y_label = "Prevalence-weighted mean abs. importance",
  show_legend = TRUE
)

# 8. Combine into 2x2 grid with unified legend
four_scatters_grid <- (p_imp_deg + p_imp_bet) / (p_deg_bet + p_clr_imp) +
  plot_layout(guides = "collect") &
  theme(legend.position = "right")

print(four_scatters_grid)

# EDGES
# -------------------------------------------------------------------------------

if (!file.exists(file.path(parent_folder, "all_seeds_edge_attributions_raw.csv"))) {
  cat("Aggregating raw edge attributions across all seeds...\n")
  
  seed_dirs <- list.dirs(parent_folder, recursive = FALSE, full.names = TRUE)
  seed_dirs <- seed_dirs[grepl("seed_", basename(seed_dirs))][1:50]

  all_seeds_edges_raw <- map_dfr(seed_dirs, function(s_dir) {
    seed_id <- basename(s_dir)
    files <- list.files(s_dir, pattern = "\\.csv$", full.names = TRUE)
    files_edges <- files[grepl("_edges", files)]
    
    map_dfr(files_edges, function(fpath) {
      fname <- basename(fpath)
      gname <- str_remove(str_remove(fname, "^integrated_gradients_edge_attributions_"), "_edges\\.csv$")
      
      df <- read_csv(fpath, show_col_types = FALSE) %>% 
        select(-any_of(c("...1", "X1")))
      
      t_label <- df$true_label[1]
      l_val   <- df$logits[1]
      c_delta <- df$convergence_delta[1]
      prob_val <- 1 / (1 + exp(-l_val))
      
      # Clean species names & collapse directed edges into undirected edge_id per seed
      df %>% 
        separate(edge, into = c("source", "target"), sep = "\\|\\|") %>% 
        mutate(
          source = if_else(str_detect(source, "s__"), str_replace(source, ".*s__", ""), source),
          source = str_replace(source, "_", " "),
          target = if_else(str_detect(target, "s__"), str_replace(target, ".*s__", ""), target),
          target = str_replace(target, "_", " ")
        ) %>% 
        rowwise() %>% 
        mutate(edge_id = paste(sort(c(source, target)), collapse = "_")) %>% 
        group_by(edge_id) %>% 
        summarise(
          attr_seed   = sum(attribution, na.rm = TRUE), # aggregating directed edges to undirected
          source      = first(source),
          target      = first(target),
          .groups         = "drop"
        ) %>% 
        mutate(
          abs_attr_seed = abs(attr_seed),
          seed              = seed_id,
          graph_name        = gname,
          true_label        = factor(t_label, levels = c(1, 0)),
          prob              = prob_val,
          logits            = l_val,
          convergence_delta = c_delta
        )
    })
  })
  
  write_csv(all_seeds_edges_raw, file.path(parent_folder, "all_seeds_edge_attributions_raw.csv"))
} else {
  cat("Aggregated edge attributions file already exists. Loading...\n")
  all_seeds_edges_raw <- read_csv(file.path(parent_folder, "all_seeds_edge_attributions_raw.csv"), show_col_types = FALSE) %>% 
    mutate(true_label = factor(true_label, levels = c(1, 0)))
}


# GRAPH EDGES SUMMARY
# ------------------------------------------------------------------------------

# Filter edge attributions to valid seeds (auprc >= 0.60)
graph_edge_summary <- all_seeds_edges_raw %>%
  filter(seed %in% valid_seeds) %>%
  select(seed, graph_name, edge_id, source, target, true_label, prob, attr_seed, abs_attr_seed) %>%
  group_by(graph_name, edge_id, source, target) %>%
  summarise(
    importance = mean(abs_attr_seed, na.rm = TRUE),
    sd_importance = sd(abs_attr_seed, na.rm = TRUE),
    net = mean(attr_seed, na.rm = TRUE),
    sd_net = sd(attr_seed, na.rm = TRUE),
    direction_consistency = abs(sum(attr_seed, na.rm = TRUE)) / sum(abs(attr_seed), na.rm = TRUE),
    n_seeds = sum(!is.na(attr_seed)),
    .groups = "drop"
  ) %>% 
  # Per-graph normalization (relative to each graph's maximum absolute edge attribution)
  group_by(graph_name) %>%
  mutate(
    max_abs = max(importance, na.rm = TRUE),
    importance_norm = importance / max_abs,
    sd_importance_norm = sd_importance / max_abs,
    net_norm = net / max_abs,
    sd_net_norm = sd_net / max_abs
  ) %>%
  ungroup() %>%
  # Join metadata
  left_join(
    HMP_metadata %>% select(subject_id, condition) %>% distinct(subject_id, .keep_all = TRUE), 
    by = c("graph_name" = "subject_id")
  ) %>%
  rename(source_old = source, target_old = target) %>%
  mutate(
    condition_broad = case_when(
      condition == "Non-IBD" ~ "Non-IBD",
      condition %in% c("UC", "CD") ~ "IBD",
      TRUE ~ "Other"
    ),
    source = clean_node_name(source_old),
    target = clean_node_name(target_old),
    edge_id_clean = paste0(source, " -- ", target)
  )

# Calculate cohort-level edge prevalence and importance ranking
edge_prevalence_df <- graph_edge_summary %>%
  group_by(edge_id_clean, source, target) %>%
  summarise(
    occurrence = n_distinct(graph_name),
    prevalence = occurrence / total_study_subjects,
    mean_importance = mean(importance_norm, na.rm = TRUE),
    mean_net = mean(net_norm, na.rm = TRUE),
    prevalence_weighted_importance = prevalence * mean_importance,
    mean_direction_consistency = mean(direction_consistency, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(prevalence_weighted_importance))

top_edges_df <- edge_prevalence_df %>%
  slice_max(prevalence_weighted_importance, n = 150) %>% 
  filter(source %in% node_prevalence_df$node_clean & target %in% node_prevalence_df$node_clean) %>% 
  select(source, target, mean_net, mean_importance, mean_direction_consistency, prevalence_weighted_importance)

max_edge_val <- max(abs(top_edges_df$mean_net), na.rm = TRUE)

# Compute mean signed (net) score per node for coloring metagraph nodes
node_net_df <- graph_node_summary %>%
  group_by(node_clean) %>%
  summarise(
    mean_net_norm = mean(net_norm, na.rm = TRUE),
    .groups = "drop"
  )

max_node_net_val <- max(abs(node_net_df$mean_net_norm), na.rm = TRUE)

backbone_graph <- as_tbl_graph(top_edges_df, directed = FALSE) %>%
  left_join(node_prevalence_df %>% rename(name = node_clean), by = "name") %>% 
  left_join(node_net_df %>% rename(name = node_clean), by = "name") %>% 
  mutate(
    node_imp = scales::rescale(prevalence_weighted_importance, to = c(0.1, 1)),
    avg_node_net_norm = mean_net_norm / max_node_net_val
  ) %>% 
  # retrieve full node names
# separate node into kingom, phylum, class, order, family, genus, species for potential future use
  left_join(graph_node_summary %>% select(node_clean, node) %>% rename(name = node_clean) %>% distinct(name, .keep_all = TRUE) %>% 
  separate(node, into = c("kingdom", "phylum", "class", "order", "family", "genus", "species"), sep = "\\|") %>% 
    # remove letter__ prefixes from taxonomic levels
    mutate(across(c(kingdom, phylum, class, order, family, genus, species), ~ str_remove(., "^[a-z]__")))
    , by = "name") %>%
  activate(edges) %>% 
  mutate(
    global_score = scales::rescale(prevalence_weighted_importance, to = c(0.1, 1)),
    avg_net_attr_norm = mean_net / max_edge_val
  ) %>% 
  activate(nodes) %>%
  mutate(component = group_components()) %>%
  filter(component == as.numeric(names(sort(table(component), decreasing = TRUE)[1])))

meta_graph_names_p <- backbone_graph %>% 
  ggraph(layout = "stress") +
  geom_edge_link(
    aes(width = global_score, alpha = global_score, color = avg_net_attr_norm),
    lineend = "round"
  ) +
  geom_node_point(
    aes(size = node_imp, color = avg_node_net_norm)
  ) +
  geom_node_text(
    aes(label = name), 
    repel = TRUE, 
    size = 3.5, 
    fontface = "italic",
    bg.color = "white", 
    bg.r = 0.15
  ) +
  scale_edge_width(range = c(0.2, 3.5), name = "Mean absolute edge importance\n(prevalence-weighted)") +
  scale_edge_alpha(range = c(0.2, 1), guide = "none") +
  scale_size_continuous(range = c(2, 10), name = "Mean absolute node importance") +
  # Heatmap-aligned colors (#D90429 = Red/IBD, white, #0F4C81 = Blue/Non-IBD)
  scale_edge_color_gradient2(
    low = "#D90429", 
    mid = "white", 
    high = "#0F4C81", 
    limits = c(-1, 1),
    name = "Mean edge signed importance"
  ) +
  scale_color_gradient2(
    low = "#D90429", 
    mid = "white", 
    high = "#0F4C81", 
    limits = c(-1, 1),
    name = "Mean node signed importance"
  ) +
  theme_graph() +
  theme(
    legend.position = "right",
    legend.title = element_text(size = 11, color = "#494949ff"),
    legend.text = element_text(size = 10, color = "#494949ff")
  )

print(meta_graph_names_p)


# Faceted Metagraph Across All Taxonomic Levels (Kingdom -> Genus)

# Define taxonomic levels
tax_levels <- c("kingdom", "phylum", "class", "order", "family", "genus")

# Extract layout & calculate per-node net edge weight from backbone_graph
node_net_weights <- backbone_graph %>%
  activate(nodes) %>%
  # define module membership based on the sum of metagraph edge weights (avg_net_attr_norm) connected to each node
  mutate(
    net_weight = centrality_degree(weights = avg_net_attr_norm, mode = "all"),
    module = if_else(net_weight >= 0, "M1", "M2")
  ) %>%
  as_tibble()

base_layout <- create_layout(backbone_graph, layout = "stress")

# Build long-format node dataframe
node_taxa_facets <- base_layout %>%
  as_tibble() %>%
  select(x, y, species_name = name, all_of(tax_levels)) %>%
  left_join(node_net_weights %>% select(name, module), by = c("species_name" = "name")) %>%
  pivot_longer(
    cols = all_of(tax_levels),
    names_to = "tax_level",
    values_to = "taxon_label"
  ) %>%
  mutate(
    tax_level = factor(str_to_title(tax_level), levels = str_to_title(tax_levels)),
    taxon_label = if_else(is.na(taxon_label) | taxon_label == "", "Unclassified", taxon_label),
    label_fontface = if_else(tax_level %in% c("Kingdom", "Phylum", "Class", "Order"), "plain", "italic")
  )

# 4. Compute Taxonomic Cohesion Metric per Facet & Module (M1 vs M2)
x_min <- min(base_layout$x)
x_max <- max(base_layout$x)
y_min <- min(base_layout$y)

# Shifted a bit more to the right (4% offset instead of 12%)
x_pos_bottom_right_offset <- x_max - 0.04 * (x_max - x_min)

# Compute cohesion as the proportion of nodes in each module that belong to the most common taxon at that level
cohesion_metrics <- node_taxa_facets %>%
  group_by(tax_level, module, taxon_label) %>%
  summarise(n = n(), .groups = "drop_last") %>%
  mutate(
    module_size = sum(n),
    purity = n / module_size
  ) %>%
  slice_max(purity, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  group_by(tax_level) %>%
  summarise(
    label_text = paste0(
      "Cohesion index\n",
      "──────────────\n",
      "M1:  ", sprintf("%5.1f%%", purity[module == "M1"] * 100), "\n",
      "M2:  ", sprintf("%5.1f%%", purity[module == "M2"] * 100)
    ),
    .groups = "drop"
  ) %>%
  mutate(
    x = x_pos_bottom_right_offset,
    y = y_min
  )

# Map edge geometry and convert scores to LIGHT BINARY RED AND BLUE
edges_df <- backbone_graph %>%
  activate(edges) %>%
  as_tibble() %>%
  select(from, to, avg_net_attr_norm) %>%
  mutate(
    edge_color = if_else(avg_net_attr_norm >= 0, "#A2C4F4", "#F4A2A2")
  )

facet_edges_df <- map_dfr(unique(node_taxa_facets$tax_level), function(lvl) {
  edges_df %>%
    mutate(
      x    = base_layout$x[from],
      y    = base_layout$y[from],
      xend = base_layout$x[to],
      yend = base_layout$y[to],
      tax_level = lvl
    )
})

# Generate palette mapping per facet level (reusing distinct colors per panel)
palette_per_facet <- node_taxa_facets %>%
  group_by(tax_level) %>%
  distinct(taxon_label) %>%
  mutate(
    color = unname(pals::cols25(n()))
  ) %>%
  ungroup()

node_taxa_facets <- node_taxa_facets %>%
  left_join(palette_per_facet, by = c("tax_level", "taxon_label"))

# Save node membership table
node_membership_table <- node_taxa_facets %>%
  distinct(species_name, module)

# write_csv(node_membership_table, "results/tables/analyses/attributions/node_module_membership_table.csv")

# Render Faceted Metagraphs using identity color scale
taxa_facets_plot <- ggplot() +
  # Edges with light binary red/blue colors
  geom_segment(
    data = facet_edges_df,
    aes(x = x, y = y, xend = xend, yend = yend),
    color = facet_edges_df$edge_color,
    linewidth = 0.35,
    alpha = 0.6
  ) +
  # Nodes colored using per-facet distinct assignments
  geom_point(
    data = node_taxa_facets,
    aes(x = x, y = y, color = color),
    size = 3.2,
    alpha = 0.95
  ) +
  # Node labels showing current taxonomic assignment per panel
  geom_text_repel(
    data = node_taxa_facets,
    aes(x = x, y = y, label = taxon_label, fontface = label_fontface),
    size = 2.8,
    bg.color = "white",
    bg.r = 0.15,
    max.overlaps = 30,
    point.padding = 0.2
  ) +
  # Styled Cohesion Card positioned in Bottom-Right
  geom_label(
    data = cohesion_metrics,
    aes(x = x, y = y, label = label_text),
    hjust = 1, vjust = 0,
    size = 2.7,
    family = "mono",
    fontface = "bold",
    color = "#2B2D42",
    fill = alpha("#FAFAFA", 0.92),
    label.padding = unit(0.45, "lines"),
    label.r = unit(0.2, "lines"),
    label.size = 0.3
  ) +
  scale_color_identity() +
  facet_wrap(~tax_level, ncol = 2) +
  theme_graph(base_family = "sans") +
  theme(
    strip.text = element_text(size = 12, face = "bold", color = "#494949ff"),
    panel.border = element_rect(color = "#494949ff", fill = NA, linewidth = 0.5),
    panel.background = element_rect(fill = "white", color = NA),
    legend.position = "none",
    plot.margin = margin(10, 10, 10, 10)
  )

print(taxa_facets_plot)



# FUNCTIONS
# ----------------------------------------------------

# Extract node names from the backbone graph for filtering functional attributes
nodes_in_graph <- backbone_graph %>% 
  activate(nodes) %>% 
  pull(name)

# Load functional attributes (GO Biological Processes) and filter to nodes present in the backbone graph
function_BP_idf <- read_csv("data/external/functional_attributes/HMP/taxa_one_hot_encoded_one_hot_bp_weighted.csv") %>% 
  mutate(node_clean = if_else(str_detect(taxa_lineage, "s__"),
                              str_replace(taxa_lineage, ".*s__", ""),
                              taxa_lineage),
         node_clean = str_replace(node_clean, "_", " ")) %>% 
  select(-taxa_lineage) %>% 
  relocate(node_clean) %>% 
  filter(node_clean %in% nodes_in_graph)

# Extract node module assignments from backbone_graph
node_modules <- backbone_graph %>%
  activate(nodes) %>%
  mutate(
    net_weight = centrality_degree(weights = avg_net_attr_norm, mode = "all"),
    module = if_else(net_weight >= 0, "M1", "M2")
  ) %>%
  as_tibble() %>%
  select(node_clean = name, module)

# Pivot IDF matrix to long format and merge module assignments
func_long <- function_BP_idf %>%
  pivot_longer(
    cols = -node_clean,
    names_to = "go_term",
    values_to = "idf_weight"
  ) %>%
  inner_join(node_modules, by = "node_clean") %>%
  mutate(has_function = if_else(idf_weight > 0, 1, 0)) # binarize presence/absence of function for Fisher's test


n_total <- nrow(node_modules)

# Perform Fisher's Exact Test per module against the rest of the graph
fisher_vs_bg <- func_long %>%
  group_by(go_term, module) %>%
  summarise(
    present_in_mod = sum(has_function),
    mod_size = n(),
    .groups = "drop"
  ) %>%
  mutate(
    total_present_net = ave(present_in_mod, go_term, FUN = sum),
    present_outside = total_present_net - present_in_mod,
    outside_size = n_total - mod_size
  ) %>%
  filter(total_present_net >= 2) %>% 
  rowwise() %>%
  mutate(
    p_val = fisher.test(
      matrix(c(present_in_mod, mod_size - present_in_mod, 
               present_outside, outside_size - present_outside), 
             nrow = 2),
      alternative = "greater"
    )$p.value
  ) %>%
  ungroup() %>%
  group_by(module) %>%
  mutate(adj_pval = p.adjust(p_val, method = "BH")) %>%
  ungroup()

# Extract top 20 terms per module (filtered by FDR < 0.05) & retain full GO codes/names
top_fisher_plot <- fisher_vs_bg %>%
  filter(adj_pval < 0.05) %>% 
  group_by(module) %>%
  slice_min(order_by = adj_pval, n = 20, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(
    neg_log10_p = -log10(adj_pval),
    dominant_in = if_else(module == "M1", "Over-represented in M1", "Over-represented in M2"),
    
    # Format GO terms: Remove "[BP]" tag but KEEP GO ID and full term name (up to 80 chars)
    go_term_formatted = go_term %>% 
      str_remove("\\[BP\\]\\s*") %>% 
      str_trunc(width = 80, side = "right"),
    
    # Reorder y-axis factor so M2 negative goes left, M1 positive goes right
    go_term_formatted = reorder(go_term_formatted, if_else(dominant_in == "Over-represented in M2", -neg_log10_p, neg_log10_p))
  )

# Render Diverging Bar Plot (Plot 1)
fun_p1 <- ggplot(top_fisher_plot, aes(
  x = if_else(dominant_in == "Over-represented in M2", -neg_log10_p, neg_log10_p), # diverging x-axis for visualization
  y = go_term_formatted, 
  fill = dominant_in
)) +
  geom_col(width = 0.7, alpha = 0.85) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  geom_vline(xintercept = -log10(0.05), linetype = "dashed", color = "black", linewidth = 0.5) +
  geom_vline(xintercept = log10(0.05), linetype = "dashed", color = "black", linewidth = 0.5) +
  scale_fill_manual(
    values = c("Over-represented in M1" = "#0F4C81", "Over-represented in M2" = "#D90429")
  ) +
  labs(
    x = expression("M2 ←  " * -log[10] * "(FDR Adjusted p-value)" * "  → M1"),
    y = "Functional attributes (GO Biological Process terms)",
    fill = NULL
  ) +
  scale_x_continuous(limits = c(-3.5, 3.5)) +
  theme_minimal(base_family = "sans") +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text = element_text(size = 9),
    legend.position = "bottom",
    legend.text = element_text(size = 9),
    axis.ticks = element_line(color = "grey50", linewidth = 0.5),
    plot.title = element_text(face = "bold", size = 12)
  )
  
print(fun_p1)


# Map Plot 1 OR terms back to species matrix
top_terms_list <- top_fisher_plot %>% pull(go_term)

species_func_matrix_clean <- func_long %>%
  filter(
    go_term %in% top_terms_list,
    has_function == 1
  ) %>%
  mutate(
    clean_species = node_clean %>% str_replace_all("_", " ") %>% str_squish(),
    
    # Format GO labels to include GO ID + Name, allowing long length
    go_clean = go_term %>% 
      str_remove("\\[BP\\]\\s*") %>% 
      str_trunc(width = 85, side = "right")
  )

# Pivot Matrix (GO Terms as Rows, Species as Columns)
wide_matrix_transposed <- species_func_matrix_clean %>%
  select(go_clean, clean_species, idf_weight) %>%
  pivot_wider(
    names_from = clean_species,
    values_from = idf_weight,
    values_fill = 0
  ) %>%
  column_to_rownames("go_clean") %>%
  as.matrix()

# Align metadata for column splits
col_metadata <- species_func_matrix_clean %>%
  distinct(clean_species, module) %>%
  slice(match(colnames(wide_matrix_transposed), clean_species))

# Setup color scale
max_val <- max(wide_matrix_transposed, na.rm = TRUE)
col_fun <- colorRamp2(
  c(0, max_val / 2, max_val),
  c("#FFFFFF", "#8E7DBE", "#3B1F52")
)

# Render Heatmap (Plot 2)
fun_p2 <- Heatmap(
  wide_matrix_transposed,
  name = "IDF Weight",
  col = col_fun,
  
  # Clustering configuration
  cluster_rows = TRUE,
  cluster_columns = TRUE,
  clustering_distance_rows = "euclidean",
  clustering_distance_columns = "euclidean",
  clustering_method_rows = "complete",
  clustering_method_columns = "complete",
  
  # Module column splitting
  column_split = col_metadata$module,
  column_title_gp = gpar(fontsize = 12, fontface = "bold"),
  
  # Typography & formatting
  show_row_names = TRUE,
  show_column_names = TRUE,
  row_names_gp = gpar(fontsize = 9), # Readable font for full GO names
  column_names_gp = gpar(fontsize = 9.5, fontface = "italic"),
  column_names_rot = 45,
  
  # Panel border & legend
  border = TRUE,
  rect_gp = gpar(col = "grey92", lwd = 0.5),
  heatmap_legend_param = list(
    title = "IDF weight",
    title_gp = gpar(fontsize = 9),
    labels_gp = gpar(fontsize = 8),
    legend_height = unit(3, "cm")
  )
)

draw(fun_p2, padding = unit(c(15, 20, 5, 50), "mm"))


save_path = "results/plots/analyses/attributions"
if (!dir.exists(save_path)) {
  dir.create(save_path, recursive = TRUE)
}

# attribution plots
fig_width <- 15
heatmap_pdf_path <- file.path(save_path, "importances_heatmap.pdf")
pdf(file = heatmap_pdf_path, width = 14, height = 11, useDingbats = FALSE)
draw(ht, merge_legend = TRUE, annotation_legend_list = list(lgd_cond_points))
decorate_annotation("Timepoints", { grid.lines(x = c(0, 1), y = unit(7, "native"), gp = gpar(col = "black", lty = 1, lwd = 1.2)) })
dev.off()

# Save correlation patchwork plot
scatter_pdf_path <- file.path(save_path, "importances_scatterplots.pdf")
ggsave(filename = scatter_pdf_path, plot = four_scatters_grid, width = 14, height = 10, units = "in", dpi = 600)

# taxonomic facets and function plots
network_pdf_path <- file.path(save_path, "importances_network.pdf")
ggsave(filename = network_pdf_path, plot = meta_graph_names_p, width = 15, height = 8, units = "in", dpi = 600, device = cairo_pdf)
network_facets_path <- file.path(save_path, "importances_network_taxonomic_facets.pdf")
ggsave(filename = network_facets_path, plot = taxa_facets_plot, width = 10, height = 12, units = "in", dpi = 600, device = cairo_pdf)
function_fisher_path <- file.path(save_path, "importances_functional_fisher.pdf")
ggsave(filename = function_fisher_path, plot = fun_p1, width = 10, height = 5, units = "in", dpi = 600)
function_heatmap_path <- file.path(save_path, "importances_functional_heatmap.pdf")
pdf(file = function_heatmap_path, width = 12, height = 7, useDingbats = FALSE)
draw(fun_p2, padding = unit(c(15, 20, 5, 50), "mm"))
dev.off()

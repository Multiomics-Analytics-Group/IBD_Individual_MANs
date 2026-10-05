library(tidyverse)
library(igraph)
library(vegan)
library(patchwork)

# Load previously processed and filtered data
#-------------------------------------------------------------------------------
IBD_composite <- readRDS("data/processed/IBD_CuratedMetagenomicData/IBD_studies_counts_filtered_composite.rds")

HMP <- IBD_composite[["HMP_2019_ibdmdb"]]

HMP_timepoint_filtered <- HMP %>% 
  group_by(subject_id) %>% 
  filter(!duplicated(days_from_first_collection)) %>%
  filter(n_distinct(days_from_first_collection) >= 7) %>%               # keep subjects with >= 7 distinct days
  arrange(subject_id, days_from_first_collection, id) %>%               # ensure deterministic ordering (earliest id per day)
  distinct(subject_id, days_from_first_collection, .keep_all = TRUE) %>% # keep one record per subject/day
  ungroup()

# Separate metadata and count table
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
  group_by(subject_id) %>% 
  mutate(ab_usage_proportion = mean(antibiotics_current_use == "yes") * 100) %>%
  ungroup() %>% 
    mutate(ab_usage_categorical = case_when(
    ab_usage_proportion == 0 ~ "0%",
    ab_usage_proportion > 0 & ab_usage_proportion <= 25 ~ "0%-25%",
    ab_usage_proportion > 25 & ab_usage_proportion <= 50 ~ "25%-50%",
    ab_usage_proportion > 50 & ab_usage_proportion <= 75 ~ "50%-75%",
    ab_usage_proportion > 75 & ab_usage_proportion <= 100 ~ "75%-100%"
  ))

# Baseline metadata (first sample per subject, without duplicates)
HMP_metadata_baseline <- HMP_metadata %>%
    filter(days_from_first_collection == 0) %>%
    distinct(subject_id, .keep_all = TRUE)

# --- CONFIGURATION ---

# Define the project root based on assumed structure
PROCESSED_DATA_PATH <- file.path("data", "processed", "pyg_datasets")

DATASETS <- c("HMP")

# --- CORE FUNCTION ---

# Function to read a GML file into an igraph object
read_gml_igraph <- function(gml_file) {
  tryCatch(
    {
      g <- read_graph(gml_file, format = "gml")
    },
    error = function(e) {
      NULL
    }

  )

  # If present, remove self-loops included for GNN training purposes
  if (!is.null(g)) {
    g_noselfloop <- igraph::simplify(g, remove.multiple = FALSE, remove.loops = TRUE)
    print(paste0(c("Removed self-loops from ", gml_file, ": ", ecount(g) - ecount(g_noselfloop), " loops removed."), collapse = ""))
    return(g_noselfloop)
  }

}

# Function to retrieve the total number of unique nodes across all loaded graphs
get_unique_nodes <- function(graphs_list) {
  unique_nodes <- c()
  for (g in graphs_list) {
    if (!is.null(g)) {
      nodes <- V(g)$name
      unique_nodes <- union(unique_nodes, nodes)
    }
  }
  return(unique_nodes)
}

# Function to compute graph metrics for a given igraph object
compute_metrics_igraph <- function(g, num_nodes_all) {

  if (is.null(g)) {
    return(tibble::tibble())
  }

  # ---- graph attributes ----
  graph_name <- igraph::graph_attr(g, "graph_name")
  true_label <- igraph::graph_attr(g, "true_label")
  true_label <- if (!is.null(true_label)) as.numeric(true_label) else NA_real_

  # ---- basic counts ----
  num_nodes <- igraph::vcount(g)
  num_edges <- igraph::ecount(g)
  isolated_nodes <- sum(igraph::degree(g, mode = "all") == 0)

  if (num_nodes == 0) {
    return(tibble::tibble(
      graph_name = graph_name,
      true_label = true_label,
      num_nodes = 0,
      num_edges = 0,
      isolated_nodes = 0
    ))
  }

  degs <- igraph::degree(g, mode = "all")

  # ---- coverage / sparsity ----
  node_coverage <- sum(degs > 0) / num_nodes_all
  density <- igraph::edge_density(g, loops = FALSE)

  # ---- degree structure ----
  mean_degree <- mean(degs)
  median_degree <- median(degs)
  max_degree <- max(degs)

  degree_cv <- if (mean_degree > 0) {
    stats::sd(degs) / mean_degree
  } else {
    NA_real_
  }

  # Gini coefficient (stable, O(n log n))
  degree_gini <- if (sum(degs) > 0) {
    s <- sort(degs)
    n <- length(s)
    sum((2 * seq_len(n) - n - 1) * s) / (n * sum(s))
  } else {
    NA_real_
  }

  # tail heaviness
  p90 <- stats::quantile(degs, 0.9, names = FALSE)
  p50 <- stats::quantile(degs, 0.5, names = FALSE)
  p90_p50_ratio <- if (p50 > 0) p90 / p50 else NA_real_

  # ---- connectivity ----
  comps <- igraph::components(g)
  n_components <- comps$no
  largest_cc_fraction <- max(comps$csize) / num_nodes

  # ---- assortativity (degree) ----
  assortativity <- assortativity_degree(g, directed = FALSE)

  # ---- global efficiency (explicit normalization) ----
  if (num_nodes > 1) {
    E(g)$dist <- 1 / abs(E(g)$weight) # higher weight = stronger connection = shorter distance
    d <- igraph::distances(g, weights =  E(g)$dist)
    inv_d <- 1 / d
    inv_d[!is.finite(inv_d)] <- 0
    global_efficiency <- sum(inv_d) / (num_nodes * (num_nodes - 1))
  } else {
    global_efficiency <- NA_real_
  }

  # ---- LCC-based metrics ----
  lcc_id <- which.max(comps$csize)
  lcc_vertices <- which(comps$membership == lcc_id)
  lcc_size <- length(lcc_vertices)

  if (lcc_size > 1) {
    g_lcc <- igraph::induced_subgraph(g, lcc_vertices)
    E(g_lcc)$weight <- abs(E(g_lcc)$weight)

    avg_path_lcc <- igraph::mean_distance(g_lcc, directed = FALSE)

    d_lcc <- igraph::distances(g_lcc)
    d_lcc <- d_lcc[is.finite(d_lcc) & d_lcc > 0]

    effective_diameter_lcc <- if (length(d_lcc) > 0) {
      stats::quantile(d_lcc, 0.9, names = FALSE)
    } else {
      NA_real_
    }
  } else {
    avg_path_lcc <- NA_real_
    effective_diameter_lcc <- NA_real_
  }

  # ---- clustering ----
  transitivity_global <- igraph::transitivity(g, type = "global")

  local_clust <- igraph::transitivity(g, type = "local")
  mean_local_clustering <- mean(local_clust, na.rm = TRUE)

  clustering_cv <- if (mean_local_clustering > 0) {
    stats::sd(local_clust, na.rm = TRUE) / mean_local_clustering
  } else {
    NA_real_
  }

  # --- edge strength ---
  total_strength <- sum(E(g)$weight, na.rm = TRUE)
  mean_strength <- if (num_nodes > 0) total_strength / num_nodes else NA_real_
  median_strength <- if (num_nodes > 0) {
    strengths <- igraph::strength(g, mode = "all", weights = E(g)$weight)
    median(strengths, na.rm = TRUE)
  } else {
    NA_real_
  }

  # positive-to-negative ratio
  positive_to_negative <- sum(E(g)$weight > 0, na.rm = TRUE) / sum(E(g)$weight < 0, na.rm = TRUE)

  # ---- output ----
  tibble::tibble(
    graph_name = graph_name,
    true_label = true_label,

    num_nodes = num_nodes,
    num_edges = num_edges,
    isolated_nodes = isolated_nodes,

    node_coverage = node_coverage,
    density = density,
    assortativity = assortativity,

    mean_degree = mean_degree,
    median_degree = median_degree,
    max_degree = max_degree,
    degree_cv = degree_cv,
    degree_gini = degree_gini,
    p90_p50_degree_ratio = p90_p50_ratio,

    n_components = n_components,
    largest_cc_fraction = largest_cc_fraction,

    global_efficiency = global_efficiency,
    avg_path_lcc = avg_path_lcc,
    effective_diameter_lcc = effective_diameter_lcc,

    transitivity_global = transitivity_global,
    mean_local_clustering = mean_local_clustering,
    clustering_cv = clustering_cv,

    total_strength = total_strength,
    mean_strength = mean_strength,
    median_strength = median_strength,
    positive_to_negative = positive_to_negative
  )
}

# --- MAIN EXECUTION LOOP ---

target_folders <- c(
  "_timepoints_self_w_functions_onehot"
)

all_metrics_df <- tibble()

for (folder_suffix in target_folders) {
  for (dataset in DATASETS) {
    
    # 1. Construct path (e.g., "CD_timepoints_self_w_functions_onehot")
    base_folder_name <- paste0(dataset, folder_suffix)
    base_path <- file.path(PROCESSED_DATA_PATH, base_folder_name)
    gml_root <- file.path(base_path, "gml")
    
    if (!dir.exists(gml_root)) {
      warning(paste("Skipping: Path not found", gml_root))
      next
    }
    
    gml_threshold_dirs <- list.dirs(gml_root, recursive = FALSE, full.names = TRUE)
    
    for (gml_dir in gml_threshold_dirs) {
      threshold_name <- basename(gml_dir)
      gml_files <- list.files(gml_dir, pattern = "\\.gml$", full.names = TRUE)
      
      if (length(gml_files) == 0) next
      
      # --- Read and Compute ---
      graphs_list <- purrr::map(gml_files, read_gml_igraph)
      unique_nodes <- get_unique_nodes(graphs_list)
      num_nodes_all <- length(unique_nodes)
      
      threshold_metrics <- purrr::map_dfr(
        graphs_list,
        compute_metrics_igraph,
        num_nodes_all = num_nodes_all
      )
      
      # 2. Add Critical Metadata
      threshold_metrics <- threshold_metrics %>%
        mutate(
          dataset = dataset, 
          threshold_raw = threshold_name,
          # This extracts 'timepoints_self', 'global_ref', etc. for your plot labels
          pipeline = str_remove(folder_suffix, "^_") 
        )
      
      all_metrics_df <- bind_rows(all_metrics_df, threshold_metrics)
    }
  }
}

# --- DATA AGGREGATION ---

final_metrics_df <- all_metrics_df %>%
  mutate(
    true_label = factor(true_label, levels = c(0, 1), labels = c("IBD", "Non-IBD")),
    threshold = str_replace(threshold_raw, "_", "."),
    threshold = as.numeric(threshold)
  ) %>% 
  mutate(
    true_label = factor(true_label, levels = c("Non-IBD", "IBD"))
  )

# --- PLOTTING (Print to R Graphics Device) ---

IBD_colors <- c("Non-IBD" = "#4C72B0", "IBD" = "indianred")
IBD_colors2 <- c(
  "Non-IBD" = "#0F4C81",  # Deep Midnight Blue (Dark/Neutral base)
  "UC"      = "#FFB703",  # Bright Amber/Yellow
  "CD"      = "#D90429"   # Vivid Crimson Red
)


# --- ANALYZE GRAPHS AND SUBJECT METADATA ---
metrics_and_metadata <- final_metrics_df %>%
  inner_join(
    HMP_metadata_baseline, by = c("graph_name" = "subject_id")
  )

# --- 1. Select graph metrics and metadata ---

graph_metrics <- c(
  "num_nodes", "isolated_nodes", "num_edges", "mean_degree", "density",
  "positive_to_negative", "assortativity",
  "global_efficiency", "p90_p50_degree_ratio"
)

metadata_vars <- c("age_category", "gender", "condition", "ab_usage_categorical", "true_label")


plot_metrics <- function(data, thrsh, target_pipeline) {

  p <- data %>% 
  filter(threshold == thrsh) %>% 
  filter(str_detect(pipeline, target_pipeline)) %>%
  select(all_of(graph_metrics), pipeline, condition, true_label, ab_usage_categorical, age_category) %>%
  mutate(condition_broad = case_when(
    condition == "Non-IBD" ~ "Non-IBD",
    TRUE ~ "IBD"
  )) %>%
  mutate(condition_broad = factor(condition_broad, levels = c("Non-IBD", "IBD"))) %>%
  pivot_longer(cols = all_of(graph_metrics), names_to = "metric", values_to = "value") %>%
  mutate(metric = case_when(
    metric == "num_nodes" ~ "Number of nodes",
    metric == "isolated_nodes" ~ "Number of isolated nodes",
    metric == "num_edges" ~ "Number of edges",
    metric == "mean_degree" ~ "Mean degree",
    metric == "density" ~ "Density",
    metric == "positive_to_negative" ~ "Positive/Negative edge ratio",
    metric == "assortativity" ~ "Degree assortativity",
    metric == "global_efficiency" ~ "Global efficiency",
    metric == "p90_p50_degree_ratio" ~ "P90/P50 degree ratio",
    TRUE ~ metric
  )) %>%
  mutate(metric = factor(metric, levels = c(
    "Number of nodes", "Number of isolated nodes",
    "Number of edges", "Mean degree",
    "Positive/Negative edge ratio", "Degree assortativity",
    "Density", "Global efficiency",
    "P90/P50 degree ratio"
  ))) %>%
  filter(!is.na(value)) %>%
  mutate(point_fill = case_when(
    ab_usage_categorical == "75%-100%" ~ "AB 75-100%",
    ab_usage_categorical == "50%-75%"  ~ "AB 50-75%",
    age_category == "senior"           ~ "Senior",
    TRUE                               ~ NA_character_
  )) %>%
  ggplot(aes(x = condition, y = value)) +
  # Boxplots by condition
  geom_boxplot(aes(fill = condition), alpha = 0.6, color = "black") +
  facet_wrap(~ metric, scales = "free_y", nrow = 1) +
  labs(
    x = "Condition",
    y = "Value",
    fill = "Condition"
  ) +
  theme_minimal() +
  scale_fill_manual(values = IBD_colors2) +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE, digits = 2)) +
  guides(fill = "none") + # hide redundant fill legend, keep shape legend
  theme(
    panel.grid = element_blank(),
    panel.border = element_rect(color = "#494949ff", fill = NA, linewidth = 0.5),
    strip.text = element_text(size = 9, face = "bold"),
    axis.text = element_text(size = 13, color = "#494949ff"),
    axis.ticks = element_line(color = "#494949ff"),
    axis.title = element_text(size = 14, color = "#494949ff"),
    axis.text.x = element_text(size = 11, color = "#494949ff"),
    legend.position = "none"
  )

  return(p)

}

stats_metrics <- function(data, thrsh, target_pipeline) {
  d_cleaned <- data %>% 
    filter(threshold == thrsh) %>% 
    filter(str_detect(pipeline, target_pipeline)) %>%
    select(all_of(graph_metrics), pipeline, condition, true_label, ab_usage_categorical, age_category) %>%
    mutate(condition_broad = case_when(
      condition == "Non-IBD" ~ "Non-IBD",
      TRUE ~ "IBD"
    )) %>%
    pivot_longer(cols = all_of(graph_metrics), names_to = "metric", values_to = "value") %>%
    mutate(metric = case_when(
      metric == "num_nodes" ~ "Number of nodes",
      metric == "isolated_nodes" ~ "Number of isolated nodes",
      metric == "num_edges" ~ "Number of edges",
      metric == "mean_degree" ~ "Mean degree",
      metric == "density" ~ "Density",
      metric == "positive_to_negative" ~ "Positive/Negative edge ratio",
      metric == "assortativity" ~ "Degree assortativity",
      metric == "global_efficiency" ~ "Global efficiency",
      metric == "p90_p50_degree_ratio" ~ "P90/P50 degree ratio",
      TRUE ~ metric
    )) %>%
    mutate(metric = factor(metric, levels = c(
      "Number of nodes", "Number of isolated nodes",
      "Number of edges", "Mean degree",
      "Positive/Negative edge ratio", "Degree assortativity",
      "Density", "Global efficiency",
      "P90/P50 degree ratio"
    ))) %>%
    filter(!is.na(value)) %>% 
    # remove metrics with zero variance (Wilcoxon will fail)
    group_by(metric) %>%
    filter(var(value) > 0) %>%
    ungroup()

  # Statistical Testing (Wilcoxon + Benjamini-Hochberg)
  stats_results <- d_cleaned %>%
    group_by(pipeline, metric) %>%
    rstatix::wilcox_test(value ~ condition_broad) %>%
    rstatix::adjust_pvalue(method = "fdr") %>%  # Benjamini-Hochberg
    rstatix::add_significance() %>%
    rstatix::add_xy_position(x = "condition_broad") # Useful for plotting later

  # Return both as a list
  return(list(data = d_cleaned, stats = stats_results))
}

result1 <- stats_metrics(metrics_and_metadata, 0.3, "timepoints_self")
result1$stats

p1_metrics <- plot_metrics(metrics_and_metadata, thrsh = 0.3, target_pipeline = "timepoints_self")


plot_pipeline_pca <- function(data, thrsh, target_pipeline, graph_metrics, IBD_colors, is_first = FALSE) {
  
  # Filter for specific pipeline
  sub_data <- data %>%
    filter(threshold == thrsh) %>% 
    filter(str_detect(pipeline, target_pipeline)) %>%
    drop_na(all_of(graph_metrics)) %>% 
    # drop single-value columns that would cause PCA to fail
    select(-where(~ n_distinct(.) <= 1))
  
  valid_cols <- graph_metrics[graph_metrics %in% colnames(sub_data)]
  
  if(nrow(sub_data) < 5) return(NULL) # Safety check
  
  # PCA Calculation
  pca_res <- prcomp(sub_data %>% select(all_of(valid_cols)), center = TRUE, scale. = TRUE)
  
  # Extract Variance
  pc_var <- round((pca_res$sdev^2) / sum(pca_res$sdev^2) * 100, 1)
  
  # Prepare Plot Data
  pca_df <- as_tibble(pca_res$x[,1:2]) %>%
    bind_cols(sub_data %>% select(-all_of(valid_cols))) %>%
    mutate(
      point_fill = case_when(
        ab_usage_categorical %in% c("75%-100%", "50%-75%") ~ "High AB",
        age_category == "senior" ~ "Senior",
        TRUE ~ NA_character_
      )
    )

  # Generate Plot
  p <- ggplot(pca_df, aes(x = PC1, y = PC2, color = true_label)) +
    geom_point(size = 3, alpha = 0.75) +
    scale_color_manual(values = IBD_colors) +
    labs(
      title = str_wrap(target_pipeline, width = 30),
      x = paste0("PC1 (", pc_var[1], "%)"),
      y = if(is_first) paste0("PC2 (", pc_var[2], "%)") else NULL
    ) +
    theme_minimal() +
    theme(
      panel.grid = element_blank(),
      panel.border = element_rect(color = "#494949ff", fill = NA, linewidth = 0.5),
      axis.text = element_text(size = 13, color = "#494949ff"),
      axis.ticks = element_line(color = "#494949ff"),
      axis.title.x = element_text(size = 14, color = "#494949ff"),
      axis.title.y = element_text(size = 14, color = "#494949ff"),
      legend.position = "none"
    )
  
  return(p)
}


p_pca1 <- plot_pipeline_pca(metrics_and_metadata, thrsh = 0.3,
  target_pipeline = "timepoints_self", graph_metrics, IBD_colors, is_first = TRUE)

plot_pipeline_dispersion <- function(data, thrsh, target_pipeline, graph_metrics, IBD_colors) {
  
  # 1. Filter and Scale (Required for meaningful Euclidean distance)
  sub_data <- data %>%
    filter(threshold == thrsh) %>% 
    filter(str_detect(pipeline, target_pipeline)) %>%
    drop_na(all_of(graph_metrics))
  
  if(nrow(sub_data) < 5) return(NULL)
  
  mat_scaled <- scale(sub_data %>% select(all_of(graph_metrics)))
  
  # Run PERMDISP
  dist_mat <- dist(mat_scaled)
  bd_res <- vegan::betadisper(dist_mat, sub_data$true_label)
  perm_test <- vegan::permutest(bd_res, permutations = 9999)
  p_val <- perm_test$tab$`Pr(>F)`[1]
  
  # 3. Prepare Plot Data
  bd_plot_df <- sub_data %>%
    mutate(
      distance = bd_res$distances
    )

  # 4. Generate Styled Plot
  p <- ggplot(bd_plot_df, aes(x = true_label, y = distance, fill = true_label)) +
    geom_boxplot(alpha = 0.8, outlier.shape = NA, color = "black") +
    # Faded points
    geom_jitter(
      width = 0.15, size = 2, shape = 21, color = "black", alpha = 0.5
    ) +
    scale_fill_manual(values = IBD_colors) +
    labs(
      title = str_wrap(target_pipeline, width = 30),
      subtitle = paste0("PERMDISP p = ", round(p_val, 4)),
      x = "Condition",
      y = "Distance to centroid"
    ) +
    theme_minimal() +
    theme(
      panel.grid = element_blank(),
      panel.border = element_rect(color = "#494949ff", fill = NA, linewidth = 0.5),
      axis.text = element_text(size = 13, color = "#494949ff"),
      axis.ticks = element_line(color = "#494949ff"),
      axis.title = element_text(size = 14, color = "#494949ff"),
      legend.position = "none"
    )
  
  return(p)
}

p1_disp <- plot_pipeline_dispersion(metrics_and_metadata, 0.3, "timepoints_self", graph_metrics, IBD_colors = IBD_colors)


# --- ADDITIONAL ANALYSES: JACCARD SIMILARITY BY TAXONOMIC LEVEL ---

graph_conditions <- HMP_metadata_baseline %>%
  select(subject_id, condition) %>%
  mutate(
    subject = as.character(subject_id),
    condition_binary = as.character(condition)
  ) %>%
  select(subject, condition_binary)

separate_taxonomic_levels <- function(df) {
  df %>%
    mutate(
      across(c(node1, node2), as.character),
      kingdom1 = str_extract(node1, "(?<=k__)[^|]+"),
      kingdom2 = str_extract(node2, "(?<=k__)[^|]+"),
      phylum1 = str_extract(node1, "(?<=p__)[^|]+"),
      phylum2 = str_extract(node2, "(?<=p__)[^|]+"),
      class1 = str_extract(node1, "(?<=c__)[^|]+"),
      class2 = str_extract(node2, "(?<=c__)[^|]+"),
      order1 = str_extract(node1, "(?<=o__)[^|]+"),
      order2 = str_extract(node2, "(?<=o__)[^|]+"),
      family1 = str_extract(node1, "(?<=f__)[^|]+"),
      family2 = str_extract(node2, "(?<=f__)[^|]+"),
      genus1 = str_extract(node1, "(?<=g__)[^|]+"),
      genus2 = str_extract(node2, "(?<=g__)[^|]+"),
      species1 = str_extract(node1, "(?<=s__).*"),
      species2 = str_extract(node2, "(?<=s__).*")
    )
}

target_folders <- c(
  "self" = "_timepoints_self_w_functions_onehot"
)

master_edge_tax_df <- tibble()

for (pipe_name in names(target_folders)) {
  folder_suffix <- target_folders[[pipe_name]]
  
  for (dataset in DATASETS) {
    base_path <- file.path(PROCESSED_DATA_PATH, paste0(dataset, folder_suffix))
    gml_root <- file.path(base_path, "gml")
    
    if (!dir.exists(gml_root)) next
    
    gml_threshold_dirs <- list.dirs(gml_root, recursive = FALSE, full.names = TRUE)
    
    for (gml_dir in gml_threshold_dirs) {
      threshold_name <- basename(gml_dir)
      # Convert "0_3" to 0.3
      current_thr <- as.numeric(str_replace(threshold_name, "_", "."))
      
      # Use 0.3 for self, 0.3 for others as per your requirement
      if (pipe_name == "self" && current_thr != 0.3) next
      if (pipe_name != "self" && current_thr != 0.3) next
      
      gml_files <- list.files(gml_dir, pattern = "\\.gml$", full.names = TRUE)
      if (length(gml_files) == 0) next
      
      message(paste("Processing edges for:", pipe_name, "Threshold:", current_thr))
      
      pipeline_edges <- purrr::map_dfr(gml_files, function(f) {
        g <- read_gml_igraph(f)
        if (is.null(g) || igraph::ecount(g) == 0) return(NULL)
        
        igraph::as_data_frame(g, what = "edges") %>%
          as_tibble() %>%
          transmute(
            node1 = as.character(from),
            node2 = as.character(to),
            weight = as.numeric(weight),
            # Extract subject ID from the graph attribute
            subject = as.character(igraph::graph_attr(g, "graph_name"))
          ) %>%
          mutate(
            tmp1 = pmin(node1, node2),
            tmp2 = pmax(node1, node2),
            node1 = tmp1, node2 = tmp2
          ) %>%
          filter(weight != 0)
      })
      
      if (nrow(pipeline_edges) > 0) {
        pipeline_tax <- separate_taxonomic_levels(pipeline_edges) %>%
          left_join(graph_conditions, by = "subject") %>%
          mutate(pipeline = pipe_name, threshold = current_thr)
        
        master_edge_tax_df <- bind_rows(master_edge_tax_df, pipeline_tax)
      }
    }
  }
}


tax_ranks <- c("species", "genus", "family", "order", "class", "phylum", "kingdom")

# Function to calculate Jaccard for either Nodes or Edges
calculate_jaccard <- function(df, type = "edge") {
  df %>%
    group_split(pipeline, condition_binary) %>%
    map_dfr(function(sub_df) {
      p_name <- unique(sub_df$pipeline)
      c_name <- unique(sub_df$condition_binary)
      
      map_dfr(tax_ranks, function(rank) {
        r1 <- paste0(rank, "1")
        r2 <- paste0(rank, "2")
        
        # Filter for valid taxa at this rank
        rank_df <- sub_df %>%
          filter(!is.na(.data[[r1]]), !is.na(.data[[r2]]), 
                 .data[[r1]] != "", .data[[r2]] != "")
        
        if (nrow(rank_df) == 0) return(NULL)

        # Generate Sets per Subject
        subject_sets <- if(type == "edge") {
          # EDGE SETS: Unique pairs
          rank_df %>%
            mutate(item = paste(.data[[r1]], .data[[r2]], sep = "...")) %>%
            split(.$subject) %>% lapply(function(x) unique(x$item))
        } else {
          # NODE SETS: Unique individual taxa
          rank_df %>%
            select(subject, !!sym(r1), !!sym(r2)) %>%
            pivot_longer(cols = -subject, values_to = "item") %>%
            split(.$subject) %>% lapply(function(x) unique(x$item))
        }
        
        subjects <- names(subject_sets)
        if (length(subjects) < 2) return(NULL)
        
        # Compute Pairwise Jaccard
        j_vals <- combn(subjects, 2, function(x) {
          s1 <- subject_sets[[x[1]]]; s2 <- subject_sets[[x[2]]]
          inter <- length(intersect(s1, s2))
          uni   <- length(union(s1, s2))
          if(uni == 0) 0 else inter / uni
        })
        
        tibble(jaccard = j_vals, tax_rank = rank, pipeline = p_name, condition = c_name, type = type)
      })
    })
}

# Run both calculations
jaccard_node_data <- calculate_jaccard(master_edge_tax_df, type = "node")
jaccard_edge_data <- calculate_jaccard(master_edge_tax_df, type = "edge")

# Combine and Format
full_jaccard_df <- bind_rows(jaccard_node_data, jaccard_edge_data) %>%
  mutate(
    tax_rank = factor(str_to_title(tax_rank), levels = str_to_title(tax_ranks)),
    type = factor(type, levels = c("node", "edge"), labels = c("Node content", "Edge content"))
  ) %>% 
  mutate(condition = if_else(condition == "Non-IBD", "Non-IBD", "IBD")) %>% 
  mutate(condition = factor(condition, levels = c("Non-IBD", "IBD"))) %>% 
   mutate(pipeline = case_when(
     pipeline == "self" ~ "Timepoints",
     TRUE ~ pipeline
   ))


type_colors <- c("Node content" = "#1b9e77", "Edge content" = "#d95f02")

# Plot jaccard similarity for the first network inference strategy only (edge filtering through a global reference)
jaccard_plot <- full_jaccard_df %>%
    filter(pipeline == "Timepoints") %>%
    ggplot(aes(x = tax_rank, y = jaccard, color = type, group = type)) +
    stat_summary(fun = median, geom = "line", linewidth = 1.1, position = position_dodge(width = 0.2)) +
    stat_summary(fun.min = function(x) quantile(x, 0.25),
                 fun.max = function(x) quantile(x, 0.75),
                 geom = "ribbon", aes(fill = type), alpha = 0.12, color = NA,
                 position = position_dodge(width = 0.2)) +
    stat_summary(fun = median, geom = "point", size = 3, position = position_dodge(width = 0.2)) +
    facet_wrap(~ condition, nrow = 1) +
    scale_color_manual(values = type_colors) +
    scale_fill_manual(values = type_colors) +
    labs(
      x = "Taxonomic rank",
      y = "Pairwise Jaccard similarity",
      color = "Content",
      fill = "Content"
    ) +
    theme_minimal() +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.border = element_rect(color = "#494949ff", fill = NA, linewidth = 0.5),
      axis.text.y = element_text(size = 12, color = "#494949ff"),
      axis.text.x = element_text(size = 10, color = "#494949ff"),
      axis.ticks = element_line(color = "#494949ff"),
      axis.title = element_text(size = 14, color = "#494949ff"),
      legend.position = "bottom",
      legend.title = element_text(size = 12, color = "#494949ff"),
      legend.text = element_text(size = 11, color = "#494949ff"),
      strip.text = element_text(size = 12, face = "bold", color = "#494949ff")
    )


# --- SAVE PLOTS TO FILES --- #

out_dir <- "results/plots/analyses/network_metrics"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

ggsave(filename = file.path(out_dir, "graph_metrics_boxplots.pdf"), plot = p1_metrics, width = 20, height = 4, dpi = 300)
ggsave(filename = file.path(out_dir, "jaccard_similarity.pdf"), plot = jaccard_plot, width = 12, height = 5, dpi = 300)
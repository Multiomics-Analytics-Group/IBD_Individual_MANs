library(tidyverse)
library(vegan)

# Load previously processed and filtered data
#-------------------------------------------------------------------------------
IBD_composite <- readRDS("data/processed/IBD_CuratedMetagenomicData/IBD_studies_counts_filtered_composite.rds")

Hall <- IBD_composite[["HallAB_2017"]]

Hall_timepoint_filtered <- Hall %>% 
  group_by(subject_id) %>% 
  filter(!duplicated(days_from_first_collection)) %>%
  filter(n_distinct(days_from_first_collection) >= 7) %>%  # keep subjects with >= 7 distinct days
  arrange(subject_id, days_from_first_collection, id) %>%  
  # Hall AB has 2 individuals with a single mislabeled timepoint (condition column). We assume that the correct condition is the majority condition
  mutate(condition = case_when(
    subject_id == "SKST006" ~ "UC", #Checked manually
    subject_id == "p9216" ~ "CD", #Checked manually
    TRUE ~ condition
  )) %>%
  distinct(subject_id, days_from_first_collection, .keep_all = TRUE) %>% # keep one record per subject/day
  # remove empty columns
  select(where(~ !all(is.na(.)))) %>%
  ungroup()

# Separate metadata and count table
metadata_cols <- names(Hall_timepoint_filtered)[!startsWith(names(Hall_timepoint_filtered), "k__")]
taxa_cols <- names(Hall_timepoint_filtered)[startsWith(names(Hall_timepoint_filtered), "k__")]

Hall_metadata <- Hall_timepoint_filtered %>%
    select(all_of(metadata_cols)) %>% 
    mutate(condition = case_when(
    condition == "healthy" ~ "Non-IBD",
    condition == "UC" ~ "UC",
    condition == "CD" ~ "CD",
    TRUE ~ condition
  )) %>% 
  mutate(condition = factor(condition, levels = c("Non-IBD", "UC", "CD")))

# Baseline metadata (first sample per subject, without duplicates)
Hall_metadata_baseline <- Hall_metadata %>%
  group_by(subject_id) %>%
  filter(days_from_first_collection == min(days_from_first_collection)) %>% # The first timepoint for HallAB is 30, not 0.
  ungroup() %>% 
  distinct(subject_id, .keep_all = TRUE)

Hall_counts <- Hall_timepoint_filtered %>%
    select(all_of(taxa_cols), id) %>%
    column_to_rownames("id")

# Transform counts to CLR and relative abundances
# --------------------------------------------------------------------------------
Hall_clr <- Hall_counts %>%
    vegan::decostand(method = "clr", pseudocount = 1)

# Aggregate samples to the subject level
# ---------------------------------------------------------------------------------------
Hall_clr_agg <- Hall_clr %>%
    as.data.frame() %>%
    rownames_to_column("id") %>%
    left_join(Hall_metadata %>%
                  select(id, subject_id),
              by = "id") %>%
    select(-id) %>%
    group_by(subject_id) %>%
    summarise(across(everything(), mean), .groups = "drop") %>%
    column_to_rownames("subject_id")

# Compute aitchison distances from aggregated CLR data
# ---------------------------------------------------------------------------------------
Hall_aitchison_dist <- vegan::vegdist(Hall_clr_agg, method = "euclidean")
Hall_aitchison_timepoint_dist <- vegan::vegdist(Hall_clr, method = "euclidean")

# Compute Within-subject and Between-subject distances for timepoint data
# ---------------------------------------------------------------------------------------
dist_mat_tp <- as.matrix(Hall_aitchison_timepoint_dist)

# Get the indices for the lower triangle (to avoid duplicates and self-comparisons)
low_tri <- which(lower.tri(dist_mat_tp), arr.ind = TRUE)

# Create a data frame of distances
# helper to build distance-comparison table from a distance matrix + lower-triangle indices
make_dist_comparison <- function(dist_mat, low_tri, metadata) {
  pos <- cbind(low_tri[, 1], low_tri[, 2])
  data.frame(
    id1 = rownames(dist_mat)[low_tri[, 1]],
    id2 = colnames(dist_mat)[low_tri[, 2]],
    distance = dist_mat[pos]
  ) %>%
    # join metadata for sample 1
    left_join(metadata %>% select(id, subject_id, condition),
              by = c("id1" = "id")) %>%
    rename(subject1 = subject_id, condition1 = condition) %>%
    # join metadata for sample 2
    left_join(metadata %>% select(id, subject_id, condition),
              by = c("id2" = "id")) %>%
    rename(subject2 = subject_id, condition2 = condition) %>%
    # within / between flag
    mutate(comparison = dplyr::if_else(subject1 == subject2, "Within-subject", "Between-subject"))
}

# apply to timepoint distance matrices
dist_comparison_species <- make_dist_comparison(dist_mat_tp, low_tri, Hall_metadata)


# Plot PCOAs and color by condition
# ---------------------------------------------------------------------------------------

IBD_colors <- c(
  "Non-IBD" = "#0F4C81",  # Deep Midnight Blue (Dark/Neutral base)
  "UC"      = "#FFB703",  # Bright Amber/Yellow
  "CD"      = "#D90429"   # Vivid Crimson Red
)

age_colors <- c(
  "Adult"     = "#7209B7",
  "Child"     = "#4EA8DE",
  "Schoolage" = "#9EF01A",
  "Senior"    = "#F15BB5"
)

comparison_colors <- c(
  "Within-subject"  = "#A2D2FF",
  "Between-subject" = "#E0E0E0"
)


Hall_metadata_baseline_agg <- Hall_metadata_baseline %>%
    distinct(subject_id, .keep_all = TRUE) %>%
    filter(subject_id %in% rownames(Hall_clr_agg)) %>%
    arrange(subject_id)

pcoa_aitchison <- cmdscale(Hall_aitchison_dist, k = 2, eig = TRUE)
pcoa_aitchison_timepoint <- cmdscale(Hall_aitchison_timepoint_dist, k = 2, eig = TRUE)

# Calculate percentage variance explained by first two PCs
pc1_pct <- round(100 * pcoa_aitchison$eig[1] / sum(pcoa_aitchison$eig), 1)
pc2_pct <- round(100 * pcoa_aitchison$eig[2] / sum(pcoa_aitchison$eig), 1)

pc1_tp_pct <- round(100 * pcoa_aitchison_timepoint$eig[1] / sum(pcoa_aitchison_timepoint$eig), 1)
pc2_tp_pct <- round(100 * pcoa_aitchison_timepoint$eig[2] / sum(pcoa_aitchison_timepoint$eig), 1)

pcoa_aitchison_df <- as.data.frame(pcoa_aitchison$points) %>%
    rownames_to_column("subject_id") %>%
    rename(PC1 = V1, PC2 = V2) %>%
    left_join(Hall_metadata_baseline_agg %>%
                  select(subject_id, condition, age_category),
              by = "subject_id") %>% 
  mutate(age_category = str_to_title(age_category))

pcoa_aitchison_tp_df <- as.data.frame(pcoa_aitchison_timepoint$points) %>%
  rownames_to_column("id") %>%
  rename(PC1 = V1, PC2 = V2) %>%
  left_join(Hall_metadata %>%
              select(id, subject_id, condition, age_category),
            by = "id") %>%
    mutate(age_category = str_to_title(age_category))


species_aitchison_plot <- ggplot(pcoa_aitchison_df, aes(x = PC1, y = PC2, color = condition)) +
    geom_point(size = 3.5, alpha = 0.7) +
    labs(
        x = paste0("PCo1 (", pc1_pct, "%)"),
        y = paste0("PCo2 (", pc2_pct, "%)"),
        color = "Condition"
    ) +
    theme_minimal() +
    scale_color_manual(values = IBD_colors) +
    theme(
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "#494949ff", fill = NA, linewidth = 0.5),
        axis.text = element_text(size = 13, color = "#494949ff"),
        axis.title = element_text(size = 14, color = "#494949ff"),
        legend.position = "none",
        legend.title.position = "top",
        legend.title = element_text(hjust = 0.5),
        legend.text = element_text(size = 8)
    )

species_aitchison_tp_plot_subject <- ggplot(pcoa_aitchison_tp_df, aes(x = PC1, y = PC2, color = condition)) +
  geom_point(size = 2.5, alpha = 0.5) +
  labs(
    x = paste0("PCo1 (", pc1_tp_pct, "%)"),
    y = paste0("PCo2 (", pc2_tp_pct, "%)"),
    color = "Condition"
  ) +
  theme_minimal() +
  scale_color_manual(values = IBD_colors) +
  theme(
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#494949ff", fill = NA, linewidth = 0.5),
    axis.text = element_text(size = 11, color = "#494949ff"),
    axis.title = element_text(size = 12, color = "#494949ff"),
    legend.position = "none",
    legend.title.position = "top",
    legend.title = element_text(hjust = 0.5),
    legend.text = element_text(size = 7)
  )

# Plot distance comparisons
# ---------------------------------------------------------------------------------------
distance_comparison_plot <- ggplot(dist_comparison_species, aes(x = comparison, y = distance, fill = comparison)) +
  geom_violin(trim = FALSE, alpha = 0.6, color = "black", width = 0.8) +
  geom_boxplot(outlier.shape = NA, alpha = 0.6, fill = "white", color = "black", width = 0.2) +
  labs(
    x = "",
    y = "Aitchison distance"
  ) +
  theme_minimal() +
  scale_fill_manual(values = comparison_colors) +
  theme(
    panel.grid = element_blank(),
    panel.border = element_rect(color = "#494949ff", fill = NA, linewidth = 0.5),
    axis.text = element_text(size = 13, color = "#494949ff"),
    axis.title = element_text(size = 14, color = "#494949ff"),
    legend.position = "none"
  )

# Compute Shannon diversities per sample and then aggregate to subject level
# ---------------------------------------------------------------------------------------
Hall_shannon <- diversity(Hall_counts, index = "shannon")

Hall_shannon_agg <- Hall_metadata %>%
    select(id, subject_id) %>%
    mutate(shannon = Hall_shannon) %>%
    group_by(subject_id) %>%
    summarise(shannon = mean(shannon), .groups = "drop")

Hall_shannon_agg <- Hall_shannon_agg %>%
    left_join(Hall_metadata_baseline_agg %>%
                  select(subject_id, condition),
              by = "subject_id")

shannon_species_plot <- ggplot(Hall_shannon_agg, aes(x = condition, y = shannon, fill = condition, color = condition)) +
  geom_jitter(width = 0.15, size = 2.5, shape = 21, color = "black", alpha = 0.8) +
  geom_boxplot(outlier.shape = NA, alpha = 0.6, color = "black") +
    labs(
        x = "Condition",
        y = "Mean Shannon Diversity"
    ) +
    theme_minimal() +
    scale_fill_manual(values = IBD_colors) +
    scale_color_manual(values = IBD_colors) +
    theme(
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "#494949ff", fill = NA, linewidth = 0.5),
        axis.text = element_text(size = 13, color = "#494949ff"),
        axis.title = element_text(size = 14, color = "#494949ff"),
        legend.position = "none"
    )


# Compute number of subjects and samples per condition
# ---------------------------------------------------------------------------------------
# Compute metadata tables for the unfiltered HMP data
Hall_metadata_unf <- Hall %>%
    select(all_of(metadata_cols)) %>% 
    mutate(condition = case_when(
    condition == "healthy" ~ "Non-IBD",
    condition == "UC" ~ "UC",
    condition == "CD" ~ "CD",
    TRUE ~ condition
  )) %>% 
  # Hall AB has 2 individuals with a single mislabeled timepoint (condition column). We assume that the correct condition is the majority condition
    mutate(condition = case_when(
      subject_id == "SKST006" ~ "UC", #Checked manually
      subject_id == "p9216" ~ "CD", #Checked manually
      TRUE ~ condition
    )) %>%
  mutate(condition = factor(condition, levels = c("Non-IBD", "UC", "CD")))

# Baseline metadata (first sample per subject, without duplicates)
Hall_metadata_baseline_unf <- Hall_metadata_unf %>%
    group_by(subject_id) %>%
    filter(days_from_first_collection == min(days_from_first_collection)) %>%
    distinct(subject_id, .keep_all = TRUE) %>% 
    ungroup()

Hall_metadata_baseline_agg_unf <- Hall_metadata_baseline_unf %>%
    distinct(subject_id, .keep_all = TRUE) %>%
    filter(subject_id %in% unique(Hall_metadata_unf$subject_id)) %>%
    arrange(subject_id)

# Compute subject counts per condition for filtered and unfiltered data
Hall_subject_counts_unf <- Hall_metadata_baseline_agg_unf %>%
    group_by(condition) %>%
    summarise(num_subjects = n_distinct(subject_id), .groups = "drop")

Hall_subject_counts <- Hall_metadata_baseline_agg %>% 
    group_by(condition) %>%
    summarise(num_subjects = n_distinct(subject_id), .groups = "drop")

Hall_subject_counts_combined <- Hall_subject_counts_unf %>%
    left_join(Hall_subject_counts, by = "condition", suffix = c("_unfiltered", "_filtered")) %>% 
  pivot_longer(cols = starts_with("num_subjects"),
               names_to = "data_type",
               values_to = "num_subjects") %>%
  mutate(data_type = recode(data_type,
                            "num_subjects_unfiltered" = "Unfiltered",
                            "num_subjects_filtered" = "Filtered")) %>% 
  mutate(data_type = factor(data_type, levels = c("Unfiltered", "Filtered"))) %>%
 mutate(color_key = paste(condition, data_type, sep = "_"))


# Compute samples per subject and attach baseline condition, then plot distribution
Hall_samples_per_subject <- Hall_metadata_unf %>%
  group_by(subject_id) %>%
  summarise(num_samples = n_distinct(id), .groups = "drop") %>%
  left_join(Hall_metadata_baseline_agg_unf %>% select(subject_id, condition),
            by = "subject_id")

samples_per_subject_plot <- ggplot(Hall_samples_per_subject, aes(x = condition, y = num_samples, fill = condition)) +
  geom_hline(yintercept = 7, linetype = "dashed", color = "black", alpha = 0.7) +
  geom_violin(trim = FALSE, alpha = 0.6, color = NA) +
  geom_boxplot(width = 0.1, outlier.shape = NA, alpha = 0.2, color = "black") +
  ggbeeswarm::geom_quasirandom(aes(fill = condition), width = 0.23, size = 1.5, shape = 21, color = "black", alpha = 0.9, groupOnX = TRUE) +
  labs(x = "Condition", y = "Samples per Subject") +
  theme_minimal() +
  scale_fill_manual(values = IBD_colors) +
  theme(
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#494949ff", fill = NA, linewidth = 0.5),
    axis.text = element_text(size = 13, color = "#494949ff"),
    axis.title = element_text(size = 14, color = "#494949ff"),
    legend.position = "none"
  )

filter_colors <- c(
  "Non-IBD_Unfiltered" = "#B4CBE2",  # Pale Ice/Cobalt (Desaturated base)
  "Non-IBD_Filtered"   = "#0F4C81",  # Deep Cobalt Blue
  
  "UC_Unfiltered"      = "#FFE9B3",  # Soft Pastel Yellow
  "UC_Filtered"        = "#FFB703",  # Bright Amber/Yellow
  
  "CD_Unfiltered"      = "#F7A8B4",  # Soft Pastel Red/Pink
  "CD_Filtered"        = "#D90429"   # Vivid Crimson Red
)


subjects_per_condition_plot <- ggplot(Hall_subject_counts_combined,
       aes(x = condition, y = num_subjects, fill = color_key,
           group = interaction(condition, data_type))) +
  geom_bar(
    stat = "identity",
    position = position_dodge2(width = 0.9, padding = 0.12),  # increased padding between groups
    color = "black",
    alpha = 0.8
  ) +
  geom_text(
    aes(label = num_subjects),
    position = position_dodge2(width = 0.9, padding = 0.12),
    vjust = -0.5,
    size = 4.5,
    color = "#494949ff"
  ) +
  scale_fill_manual(values = filter_colors) +
  labs(
    x = "Condition",
    y = "Number of Subjects"
  ) +
  theme_minimal() +
  theme(
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#494949ff", fill = NA, linewidth = 0.5),
    axis.text = element_text(size = 13, color = "#494949ff"),
    axis.title = element_text(size = 14, color = "#494949ff"),
    legend.position = "none"
  )


# Plot overall subject metadata panel
# ---------------------------------------------------------------------------------------
subject_panel_data <- Hall_metadata_baseline_agg %>% 
  select(subject_id, condition, age_category) %>% 
  mutate(age_category = str_to_title(age_category)) %>%
  pivot_longer(cols = -subject_id,
               names_to = "variable",
               values_to = "value")

color_lookup <- c(
  IBD_colors,
  age_colors)

subject_panel_data <- subject_panel_data %>%
  mutate(color = color_lookup[value]) %>% 
  mutate(variable = case_when(
    variable == "condition" ~ "Condition",
    variable == "age_category" ~ "Age category")) %>% 
  mutate(variable = factor(variable,
                           levels = c("Condition", "Age category"))) %>% 
  # order subjects by condition
  left_join(Hall_metadata_baseline_agg %>%
              select(subject_id, condition),
            by = "subject_id") %>%
  arrange(condition, subject_id) %>%
  mutate(subject_id = factor(subject_id, levels = unique(subject_id))) %>%
  select(-condition)


subject_panel_plot <- ggplot(subject_panel_data, aes(x = subject_id, y = 1, fill = value)) +
  geom_tile() +
  facet_grid(variable ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_fill_manual(values = color_lookup) +
  labs(x = "Subject ID", y = NULL, fill = NULL) +
  theme_minimal() +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid = element_blank(),
    strip.text.y = element_text(size = 14, color = "#494949ff"),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 9),
    legend.position = "none"
  )

  # create directory, save plots, and return tibble of saved paths
  out_dir <- "results/plots/analyses/diversity"
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

  plot_list <- list(
    species_aitchison_plot = species_aitchison_plot,
    species_aitchison_tp_plot_subject = species_aitchison_tp_plot_subject,
    distance_comparison_plot = distance_comparison_plot,
    shannon_species_plot = shannon_species_plot,
    samples_per_subject_plot = samples_per_subject_plot,
    subjects_per_condition_plot = subjects_per_condition_plot
  )

  saved <- purrr::imap_dfr(plot_list, ~{
    fname <- paste0(.y, "_Hall.pdf")
    path <- file.path(out_dir, fname)
    ggsave(filename = path, plot = .x, width = 7, height = 6, dpi = 300)
    tibble::tibble(plot = .y, path = path)
  })

# save panel plot separately
panel_path <- file.path(out_dir, "subject_metadata_panel_Hall.pdf")
ggsave(filename = panel_path, plot = subject_panel_plot, width = 14, height = 6, dpi = 300)

saved <- saved %>%
  add_row(plot = "subject_metadata_panel_plot", path = panel_path)

# SparCC associations among metagraph Bacteroidales, non-IBD vs IBD (Suppl. Fig. 6)
.root <- getwd(); while (!file.exists(file.path(.root, "code", "common", "paths.R")) && dirname(.root) != .root) .root <- dirname(.root)
source(file.path(.root, "code", "common", "paths.R"))
suppressMessages({ library(tidyverse); library(igraph) })
options(width = 200)

OUT_DIR <- RES_BACTEROIDALES
FIG_DIR <- FIG_BACTEROIDALES
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
M_TABLE <- file.path(RES_ABLATION, "metagraph_species.csv")

group_colors <- c("Non-IBD" = "#0F4C81", "IBD" = "#D90429")
clean_node_name <- function(x) x %>% str_remove("^.*s__") %>% str_replace_all("_", " ")
cohort_labels <- as_labeller(c(HMP = "HMP2/iHMP, discovery dataset", Hall = "Hall et al. (2017), validation dataset"))

theme_paper <- theme_minimal() +
  theme(
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#494949ff", fill = NA, linewidth = 0.5),
    axis.text = element_text(size = 13, color = "#494949ff"),
    axis.title = element_text(size = 14, color = "#494949ff"),
    strip.text = element_text(size = 13, color = "#494949ff"),
    legend.position = "none"
  )

# Metadata and number of samples per individual
IBD_composite <- readRDS(file.path(DATA_ABUNDANCE, "IBD_studies_counts_filtered_composite.rds"))
subject_info <- function(study) {
  IBD_composite[[study]] %>%
    group_by(subject_id) %>%
    filter(!duplicated(days_from_first_collection)) %>%
    summarise(n_samples = n_distinct(days_from_first_collection), condition = first(condition), .groups = "drop") %>%
    mutate(group = factor(if_else(condition == "healthy", "Non-IBD", "IBD"), levels = c("Non-IBD", "IBD")))
}

# Thresholded networks (model input)
read_networks <- function(cohort) {
  dir <- file.path(DATA_PYG, paste0(cohort, "_timepoints_self_w_functions_onehot"), "gml", "0_3")
  files <- list.files(dir, pattern = "\\.gml$", full.names = TRUE)
  setNames(map(files, ~ read_graph(.x, format = "gml")), str_remove(basename(files), "\\.gml$"))
}
networks <- list(HMP = read_networks("HMP"), Hall = read_networks("Hall"))

# Metagraph Bacteroidales, from the lineage names of the network nodes
m_species <- read_csv(M_TABLE, show_col_types = FALSE) %>% pull(species_name)
lineages <- unique(unlist(map(c(networks$HMP, networks$Hall), ~ V(.x)$name)))
bact <- tibble(lineage = lineages, species = clean_node_name(lineages),
               order = str_extract(lineages, "o__[^|]+") %>% str_remove("o__")) %>%
  filter(species %in% m_species, order == "Bacteroidales") %>% distinct(species) %>% pull(species) %>% sort()
message("Metagraph Bacteroidales: ", length(bact))

# Every pair of metagraph Bacteroidales present in a network, with its edge weight (absent = 0)
pairs_long <- function(nets, cohort) {
  imap_dfr(nets, function(g, subject) {
    sp <- clean_node_name(V(g)$name)
    present <- sort(intersect(sp, bact))
    if (length(present) < 2) return(NULL)
    all_pairs <- as_tibble(t(combn(present, 2)), .name_repair = ~ c("species_a", "species_b"))
    el <- as_edgelist(g, names = FALSE)
    edges <- tibble(a = sp[el[, 1]], b = sp[el[, 2]], weight = E(g)$weight) %>%
      filter(a %in% bact, b %in% bact) %>%
      transmute(species_a = pmin(a, b), species_b = pmax(a, b), weight)
    all_pairs %>% left_join(edges, by = c("species_a", "species_b")) %>%
      mutate(edge_present = !is.na(weight), weight = replace_na(weight, 0),
             cohort = cohort, subject_id = subject)
  })
}
pairs <- bind_rows(pairs_long(networks$HMP, "HMP"), pairs_long(networks$Hall, "Hall")) %>%
  left_join(bind_rows(subject_info("HMP_2019_ibdmdb") %>% mutate(cohort = "HMP"),
                      subject_info("HallAB_2017") %>% mutate(cohort = "Hall")),
            by = c("cohort", "subject_id")) %>%
  mutate(cohort = factor(cohort, levels = c("HMP", "Hall")), pair = paste(species_a, "-", species_b))
stopifnot(!any(is.na(pairs$group)), !any(duplicated(pairs[c("cohort", "subject_id", "pair")])))

# 1. Per individual: mean SparCC weight over all pairs (absent edges = 0), IBD vs non-IBD
# ------------------------------------------------------------------------------
per_individual <- pairs %>%
  group_by(cohort, subject_id, group) %>%
  summarise(n_species = n_distinct(c(species_a, species_b)), n_pairs = n(),
            mean_weight_all_pairs = mean(weight), .groups = "drop")
write_csv(per_individual, file.path(OUT_DIR, "per_individual_mean_sparcc.csv"))

individual_tests <- per_individual %>%
  group_by(cohort) %>%
  summarise(n_nonIBD = sum(group == "Non-IBD"), n_IBD = sum(group == "IBD"),
            mean_nonIBD = mean(mean_weight_all_pairs[group == "Non-IBD"]),
            sd_nonIBD = sd(mean_weight_all_pairs[group == "Non-IBD"]),
            mean_IBD = mean(mean_weight_all_pairs[group == "IBD"]),
            sd_IBD = sd(mean_weight_all_pairs[group == "IBD"]),
            p = wilcox.test(mean_weight_all_pairs[group == "Non-IBD"], mean_weight_all_pairs[group == "IBD"],
                            exact = FALSE)$p.value,  # two-sided Mann-Whitney U test
            .groups = "drop")
write_csv(individual_tests, file.path(OUT_DIR, "per_individual_test.csv"))
cat("\n=== Mean SparCC weight among metagraph Bacteroidales, per individual ===\n")
print(individual_tests %>% mutate(across(where(is.double), ~ signif(.x, 3))))

p_individual <- ggplot(per_individual, aes(x = group, y = mean_weight_all_pairs, fill = group)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.6, color = "black", width = 0.5) +
  geom_jitter(width = 0.12, size = 2.3, shape = 21, color = "black", alpha = 0.8) +
  geom_text(data = individual_tests %>% mutate(label = if_else(p < 0.05, "*", "n.s.")),  # *, p < 0.05
            aes(x = 1.5, y = Inf, label = label), inherit.aes = FALSE, vjust = 1.2, size = 6, color = "#494949ff") +
  facet_wrap(~ cohort, labeller = cohort_labels) +
  scale_fill_manual(values = group_colors) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.3))) +
  labs(x = "Condition", y = "Mean SparCC weight among\nmetagraph Bacteroidales") +
  theme_paper +
  theme(panel.grid = element_blank())

# 2. Per pair: mean SparCC weight per group and difference (descriptive, no tests)
# ------------------------------------------------------------------------------
pair_differences <- pairs %>%
  group_by(cohort, species_a, species_b, pair) %>%
  filter(sum(group == "Non-IBD") >= 3, sum(group == "IBD") >= 3) %>%  # at least 3 individuals per group
  summarise(n_nonIBD = sum(group == "Non-IBD"), n_IBD = sum(group == "IBD"),
            mean_nonIBD = mean(weight[group == "Non-IBD"]), mean_IBD = mean(weight[group == "IBD"]),
            diff_nonIBD_minus_IBD = mean_nonIBD - mean_IBD, .groups = "drop")
write_csv(pair_differences, file.path(OUT_DIR, "per_pair_differences.csv"))
cat("\n=== Pairs more positive in non-IBD (descriptive) ===\n")
print(pair_differences %>% group_by(cohort) %>%
        summarise(n_pairs = n(), higher_in_nonIBD = sum(diff_nonIBD_minus_IBD > 0), .groups = "drop"))

heat <- expand_grid(cohort = factor(c("HMP", "Hall"), levels = c("HMP", "Hall")), species_a = bact, species_b = bact) %>%
  # the diagonal (species with itself) is not evaluated either, so it is also marked with an X
  left_join(pair_differences %>%
              bind_rows(pair_differences %>% rename(species_a = species_b, species_b = species_a)) %>%
              select(cohort, species_a, species_b, diff_nonIBD_minus_IBD),
            by = c("cohort", "species_a", "species_b")) %>%  # NA = pair not evaluated
  mutate(species_a = factor(species_a, levels = bact), species_b = factor(species_b, levels = rev(bact)))
lim <- max(abs(heat$diff_nonIBD_minus_IBD), na.rm = TRUE)
p_heat <- ggplot(heat, aes(x = species_a, y = species_b, fill = diff_nonIBD_minus_IBD)) +
  geom_tile(color = "white") +
  # Pairs not evaluated: white cell with an X
  geom_text(data = ~ filter(.x, is.na(diff_nonIBD_minus_IBD)), label = "\u00d7", size = 3.2, color = "#494949ff") +
  facet_wrap(~ cohort, labeller = cohort_labels) +
  # Same purple-green scale as the E. coli neighbour SparCC heatmap (Fig. 5D), distinct from importance colours
  scale_fill_gradient2(low = "#7b3294", mid = "#f7f7f7", high = "#008837", limits = c(-lim, lim),
                       na.value = "white",  # pairs not evaluated (< 3 individuals in a condition), marked with an X
                       name = "Mean SparCC weight\nnon-IBD minus IBD") +
  labs(x = NULL, y = NULL, caption = "Pairs without an edge counted as 0. \u00d7: not evaluated.") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 60, hjust = 1, face = "italic", size = 9, color = "#494949ff"),
        axis.text.y = element_text(face = "italic", size = 9, color = "#494949ff"),
        strip.text = element_text(size = 13, color = "#494949ff"),
        panel.grid = element_blank(), legend.position = "bottom",
        legend.title = element_text(size = 10, color = "#494949ff"), legend.key.width = unit(1.2, "cm"))

# Save both figures at the same width with aligned left/right layout areas, so the two cohort
# panels sit at the same horizontal positions and the files can be stacked vertically
FIG_WIDTH <- 10
d_ind  <- patchwork::get_dim(p_individual)
d_heat <- patchwork::get_dim(p_heat)
lr <- list(l = pmax(d_ind$l, d_heat$l), r = pmax(d_ind$r, d_heat$r))
d_ind[c("l", "r")]  <- lr
d_heat[c("l", "r")] <- lr
ggsave(file.path(FIG_DIR, "per_individual_mean_edge_weight.pdf"), patchwork::set_dim(p_individual, d_ind),
       width = FIG_WIDTH, height = 5)
ggsave(file.path(FIG_DIR, "per_pair_difference_heatmap.pdf"), patchwork::set_dim(p_heat, d_heat),
       width = FIG_WIDTH, height = 7.5)

message("\nTables in ", OUT_DIR, "\nFigures in ", FIG_DIR)

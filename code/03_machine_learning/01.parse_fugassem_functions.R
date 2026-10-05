library(tidyverse)

# Load functional annotations from FugassEM paper
# -----------------------------------------------
fugassem_functions <- read_tsv("data/external/HMP2_fugassem_GO.func_assignment.tsv")

# Load previously generated composite tables for HMP and HallAB cohorts (discovery and validation cohorts, respectively)
# ------------------------------------------------
composite_HMP <- read_csv("data/processed/IBD_CuratedMetagenomicData/HMP_counts_filtered_composite.csv")

# Clean species names in fugassem to match composite datasets
# ---------------------------------------------------------------
fugassem <- fugassem_functions %>%
    # aggregate taxa lineage to species level, cleaning species names
    mutate(
        aggregated = if_else(
            taxa_rank == "Terminal",
            str_remove(taxa_lineage, "\\|t__.*"),
            taxa_lineage
        ),

        # clean only the portion after |s__
        aggregated = str_replace(
            aggregated,
            "(?<=\\|s__)(.*)$",
            \(x) {
                x %>%
                    # replace colons with underscores
                    str_replace_all(":", "_") %>%
                    # remove dot from sp.
                    str_replace_all("sp\\.", "sp") %>%
                    # replace dots with underscores (except between genus/species words)
                    str_replace_all("(?<=\\w)\\.(?=\\w)", "_") %>%
                    # remove [ ] brackets
                    str_replace_all("\\[|\\]", "") %>%
                    # trim trailing underscores
                    str_replace_all("_+$", "")
            }
        )
    )


# Identify taxa in composite datasets that have functional annotations in FugassEM
# ------------------------------------------------

# unique taxa with functions assigned
taxa_w_functions <- unique(fugassem$aggregated)

# column names that start with "k__" are taxa in the composite IBD datasets
composite_taxa_HMP <- colnames(composite_HMP)[-1]
composite_taxa_HMP <- composite_taxa_HMP[grepl("^k__", composite_taxa_HMP)]


# Extract species portion (everything from "|s__" onward)
get_species <- function(x) {
    str_extract(x, "\\|s__.*$")
}

# find overlapping species identifiers
functional_taxa_HMP <- intersect(
    get_species(taxa_w_functions),
    get_species(composite_taxa_HMP)
)

# find species in IBD dataset without functions assigned
no_functions_HMP <- setdiff(
    get_species(composite_taxa_HMP),
    get_species(taxa_w_functions)
)


# Filter fugassem to only include taxa with functions assigned and present in composite datasets
# -----------------------------------------------
filter_fugassem_for_composite <- function(fugassem_tbl, composite_taxa) {
  species_in_composite <- get_species(composite_taxa)
  taxa_w_functions <- unique(fugassem_tbl$aggregated)
  functional_taxa <- intersect(get_species(taxa_w_functions), species_in_composite)

  fugassem_tbl %>%
    filter(str_extract(aggregated, "\\|s__.*$") %in% functional_taxa) %>%
    select(taxon, func, category, UniRef90, aggregated) %>%
    mutate(
      taxa_lineage = composite_taxa[
        match(str_extract(aggregated, "\\|s__.*$"), species_in_composite)
      ]
    )
}

fugassem_filtered_HMP <- filter_fugassem_for_composite(fugassem, composite_taxa_HMP)

# Compute IDF and TF-IDF metrics for each function across taxa
# this is going to be used to shrink housekeeping functions
# -----------------------------------------------
compute_tf_idf <- function(fugassem_table) {
    res <- fugassem_table %>%
        mutate(func_simple = str_extract(func, "GO:\\d+")) %>%
        select(taxa_lineage, category, func) %>%
        add_count(taxa_lineage, func, name = "func_count") %>%
        distinct(taxa_lineage, func, .keep_all = TRUE)

    total_taxa <- n_distinct(res$taxa_lineage)

    res %>%
        group_by(func) %>%
        # Compute Inverse Document Frequency (IDF)
        mutate(
            n_taxa = n_distinct(taxa_lineage),
            idf = log((total_taxa + 1) / (n_taxa + 1))
        ) %>%
        ungroup() %>%
        group_by(taxa_lineage) %>%
        # Compute Term Frequency (TF)
        mutate(
            total_funcs = sum(func_count),
            tf = func_count / total_funcs
        ) %>%
        ungroup() %>%
        # Compute TF-IDF
        mutate(tf_idf = tf * idf)
}

tf_idf_metrics_HMP <- compute_tf_idf(fugassem_filtered_HMP)


# Generate one-hot encoded vectors for each taxon based on the presence/absence of functions.
# -----------------------------------------------
build_one_hot_sets <- function(tf) {
  one_hot_bp <- tf %>% 
    filter(category == "BP") %>% 
    select(taxa_lineage, func) %>% 
    mutate(present = 1) %>% 
    tidyr::pivot_wider(
      id_cols = taxa_lineage,
      names_from = func,
      values_from = present,
      values_fill = 0
    )

  one_hot_mf <- tf %>% 
    filter(category == "MF") %>% 
    select(taxa_lineage, func) %>% 
    mutate(present = 1) %>% 
    tidyr::pivot_wider(
      id_cols = taxa_lineage,
      names_from = func,
      values_from = present,
      values_fill = 0
    )

  one_hot_combined <- tf %>% 
    filter(category %in% c("BP", "MF")) %>% 
    select(taxa_lineage, func) %>% 
    mutate(present = 1) %>% 
    tidyr::pivot_wider(
      id_cols = taxa_lineage,
      names_from = func,
      values_from = present,
      values_fill = 0
    )

  one_hot_bp_weighted <- tf %>% 
    filter(category == "BP") %>% 
    select(taxa_lineage, func, idf) %>% 
    tidyr::pivot_wider(
      id_cols = taxa_lineage,
      names_from = func,
      values_from = idf,
      values_fill = 0
    )

  one_hot_mf_weighted <- tf %>% 
    filter(category == "MF") %>% 
    select(taxa_lineage, func, idf) %>% 
    tidyr::pivot_wider(
      id_cols = taxa_lineage,
      names_from = func,
      values_from = idf,
      values_fill = 0
    )

  one_hot_combined_weighted <- tf %>% 
    filter(category %in% c("BP", "MF")) %>% 
    select(taxa_lineage, func, idf) %>% 
    tidyr::pivot_wider(
      id_cols = taxa_lineage,
      names_from = func,
      values_from = idf,
      values_fill = 0
    )

  one_hot_bp_weighted_tf_idf <- tf %>% 
    filter(category == "BP") %>% 
    select(taxa_lineage, func, tf_idf) %>% 
    tidyr::pivot_wider(
      id_cols = taxa_lineage,
      names_from = func,
      values_from = tf_idf,
      values_fill = 0
    )

  one_hot_mf_weighted_tf_idf <- tf %>% 
    filter(category == "MF") %>% 
    select(taxa_lineage, func, tf_idf) %>% 
    tidyr::pivot_wider(
      id_cols = taxa_lineage,
      names_from = func,
      values_from = tf_idf,
      values_fill = 0
    )

  one_hot_combined_weighted_tf_idf <- tf %>% 
    filter(category %in% c("BP", "MF")) %>% 
    select(taxa_lineage, func, tf_idf) %>% 
    tidyr::pivot_wider(
      id_cols = taxa_lineage,
      names_from = func,
      values_from = tf_idf,
      values_fill = 0
    )

  list(
    one_hot_bp = one_hot_bp,
    one_hot_mf = one_hot_mf,
    one_hot_combined = one_hot_combined,
    one_hot_bp_weighted = one_hot_bp_weighted,
    one_hot_mf_weighted = one_hot_mf_weighted,
    one_hot_combined_weighted = one_hot_combined_weighted,
    one_hot_bp_weighted_tf_idf = one_hot_bp_weighted_tf_idf,
    one_hot_mf_weighted_tf_idf = one_hot_mf_weighted_tf_idf,
    one_hot_combined_weighted_tf_idf = one_hot_combined_weighted_tf_idf
  )
}

# Build for both datasets and keep in a named list
one_hot_sets <- list(
  HMP  = build_one_hot_sets(tf_idf_metrics_HMP)
)

# Save weighted and concatenated embeddings to CSV files
save_path <- "data/external/functional_attributes"
dir.create(save_path, showWarnings = FALSE, recursive = TRUE)

for (dataset_name in names(one_hot_sets)) {
  dataset_dir <- file.path(save_path, dataset_name)
  dir.create(dataset_dir, showWarnings = FALSE, recursive = TRUE)

  dataset <- one_hot_sets[[dataset_name]]
  for (weight_type in names(dataset)) {
    out_path <- file.path(dataset_dir, paste0("taxa_one_hot_encoded_", weight_type, ".csv"))
    readr::write_csv(dataset[[weight_type]], out_path)
  }
}
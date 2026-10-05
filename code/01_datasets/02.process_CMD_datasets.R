library(tidyverse)

# Load previously saved datasets
#-------------------------------------------------------------------------------
load_path <- "data/raw/IBD_CuratedMetagenomicData/"
IBD_CMD_counts <- readRDS(paste0(load_path, "IBD_studies_counts_raw.rds"))
IBD_CMD_counts_baseline <- readRDS(paste0(load_path, "IBD_studies_counts_raw_baseline.rds"))
IBD_metadata <- readRDS(paste0(load_path, "IBD_studies_metadata.rds"))
IBD_metadata_baseline <- readRDS(paste0(load_path, "IBD_studies_metadata_baseline.rds"))

# Apply taxonomic prevalence filter
#-------------------------------------------------------------------------------
# Prevalence filtering function
filter_by_prevalence <- function(df, prevalence_threshold = 0.1, dataset = NULL) {
    # Compute prevalence for each species
    prevalence <- df %>%
        select(-dataset) %>%
        pivot_longer(-id, names_to = "species", values_to = "reads") %>%
        group_by(species) %>%
        summarize(prev = sum(reads > 0) / n(), .groups = "drop")

    keep_species <- prevalence %>%
        filter(prev >= prevalence_threshold) %>%
        pull(species) %>%
        unique()

    print(paste0("Total number of species before prevalence filtering: ", ncol(df) - 2)) # Subtract 2 for id and dataset columns
    print(paste0("Number of species retained after prevalence filtering: ", length(keep_species)))

    filtered_table <- df %>%
        select(id, dataset, all_of(keep_species)) %>%
        #  Remove samples with zero counts across all retained species
        filter(rowSums(select(., -id, -dataset)) > 0)

    print(paste0("Total number of samples before prevalence filtering: ", nrow(df)))
    print(paste0("Number of non-zero samples retained after prevalence filtering: ", nrow(filtered_table)))

    return(filtered_table)
}


# Apply the filter individually to each dataset in the list
filtered_IBD_CMD <- list()
for (dataset in names(IBD_CMD_counts)) {
  print(paste0("Filtering dataset: ", dataset))
    filtered_IBD_CMD[[dataset]] <- filter_by_prevalence(
        IBD_CMD_counts[[dataset]],
        prevalence_threshold = 0.05, # 5% prevalence threshold
        dataset = dataset
    )
}

# Apply the same filtering to the baseline datasets
filtered_IBD_CMD_baseline <- list()
for (dataset in names(IBD_CMD_counts_baseline)) {
    filtered_IBD_CMD_baseline[[dataset]] <- filter_by_prevalence(
        IBD_CMD_counts_baseline[[dataset]],
        prevalence_threshold = 0.05, # 5% prevalence threshold
        dataset = paste0(dataset, "_baseline")
    )
}

# Create composite tables
filtered_IBD_CMD_composite <- map2(filtered_IBD_CMD, IBD_metadata, ~ {
    inner_join(.x, .y, by = "id")
})

filtered_IBD_CMD_baseline_composite <- map2(filtered_IBD_CMD_baseline, IBD_metadata_baseline, ~ {
    inner_join(.x, .y, by = "id")
})

# Save filtered datasets
save_path2 <- "data/processed/IBD_CuratedMetagenomicData/"
dir.create(save_path2, showWarnings = FALSE, recursive = TRUE)
saveRDS(filtered_IBD_CMD, paste0(save_path2, "IBD_studies_counts_filtered.rds"))
saveRDS(filtered_IBD_CMD_baseline, paste0(save_path2, "IBD_studies_counts_filtered_baseline.rds"))
saveRDS(filtered_IBD_CMD_composite, paste0(save_path2, "IBD_studies_counts_filtered_composite.rds"))
saveRDS(filtered_IBD_CMD_baseline_composite, paste0(save_path2, "IBD_studies_counts_filtered_baseline_composite.rds"))
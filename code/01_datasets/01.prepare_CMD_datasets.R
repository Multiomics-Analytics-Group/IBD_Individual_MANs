library(tidyverse)

# Download IBD datasets from CuratedMetagenomicData
#-------------------------------------------------------------------------------
get_data_from_CMD <- function(datasets, data_type, counts) {
    data <- curatedMetagenomicData::sampleMetadata %>%
        filter(study_name %in% datasets) %>%
        select(where(~ !all(is.na(.x)))) %>%
        curatedMetagenomicData::returnSamples(data_type, counts = counts)

    data
}

datasets <- c("HMP_2019_ibdmdb", "HallAB_2017", "IjazUZ_2017", "NielsenHB_2014", "VilaAV_2018")

# Get taxonomic data
IBD_CMD <- get_data_from_CMD(
    datasets = datasets,
    data_type = "relative_abundance",
    counts = TRUE
)

# Retrieve and process metadata
#-------------------------------------------------------------------------------
# Retrieve metadata including all timepoints for longitudinal datasets
IBD_CMD_metadata <- as.data.frame(colData(IBD_CMD)) %>%
    rownames_to_column(var = "id") %>%
    group_by(study_name) %>%
    mutate(n_samples = n()) %>%
    ungroup() %>%
    # Filter for datasets with at least 100 samples
    filter(n_samples >= 100) %>%
    mutate(disease_subtype = if_else(is.na(disease_subtype), "healthy", disease_subtype)) %>%
    filter(disease_subtype != "undetermined_colitis") %>%
    rename(
        "study_name" = "dataset",
        "disease_subtype" = "condition"
    ) %>%
    distinct(id, .keep_all = TRUE)

# Metadata of the baseline samples.
IBD_CMD_metadata_baseline <- IBD_CMD_metadata %>%
    group_by(dataset, subject_id) %>%
    filter(
        # VilaAV_2018 is a cross-sectional study, so keep all samples
        dataset == "VilaAV_2018" |
            days_from_first_collection == 0 |
            days_from_first_collection == min(days_from_first_collection) # HallAB starts at 30
    ) %>%
    ungroup() %>%
    distinct(subject_id, .keep_all = TRUE)

# Retrieve count data and pathway abundance data for all timepoints
#--------------------------------------------------------------------------------
# Counts
IBD_CMD_counts <- as.data.frame(t(assay(IBD_CMD, "relative_abundance"))) %>%
    rownames_to_column(var = "id") %>%
    filter(id %in% IBD_CMD_metadata$id) %>%
    inner_join(., IBD_CMD_metadata %>% select(id, dataset), by = "id") %>%
    group_split(dataset, .keep = TRUE) %>%
    lapply(function(x) {
        x %>%
            as.data.frame() %>%
            # remove columns with all zeros
            select(where(~ !all(. == 0)))
    })

IBD_CMD_counts_baseline <- as.data.frame(t(assay(IBD_CMD, "relative_abundance"))) %>%
    rownames_to_column(var = "id") %>%
    filter(id %in% IBD_CMD_metadata_baseline$id) %>%
    inner_join(., IBD_CMD_metadata_baseline %>% select(id, dataset), by = "id") %>%
    group_split(dataset, .keep = TRUE) %>%
    lapply(function(x) {
        x %>%
            as.data.frame() %>%
            # remove columns with all zeros
            select(where(~ !all(. == 0)))
    })

# Assign dataset names to each element of the lists
names(IBD_CMD_counts) <- sapply(IBD_CMD_counts, function(x) unique(x$dataset))
names(IBD_CMD_counts_baseline) <- sapply(IBD_CMD_counts_baseline, function(x) unique(x$dataset))

# Separate metadata per dataset
IBD_metadata <- IBD_CMD_metadata %>%
    group_split(dataset, .keep = TRUE)

IBD_metadata_baseline <- IBD_CMD_metadata_baseline %>%
    group_split(dataset, .keep = TRUE)

names(IBD_metadata) <- sapply(IBD_metadata, function(x) unique(x$dataset))
names(IBD_metadata_baseline) <- sapply(IBD_metadata_baseline, function(x) unique(x$dataset))

# Save raw files
save_path <- "data/raw/IBD_CuratedMetagenomicData/"
dir.create(save_path, recursive = TRUE, showWarnings = FALSE)

saveRDS(IBD_CMD_counts, paste0(save_path, "IBD_studies_counts_raw.rds"))
saveRDS(IBD_CMD_counts_baseline, paste0(save_path, "IBD_studies_counts_raw_baseline.rds"))
saveRDS(IBD_metadata, paste0(save_path, "IBD_studies_metadata.rds"))
saveRDS(IBD_metadata_baseline, paste0(save_path, "IBD_studies_metadata_baseline.rds"))

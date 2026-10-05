# Locate the project root (from any working directory inside the repo) and load shared paths
.root <- getwd(); while (!file.exists(file.path(.root, "code", "common", "paths.R")) && dirname(.root) != .root) .root <- dirname(.root)
source(file.path(.root, "code", "common", "paths.R"))

library(tidyverse)
source(file.path(PROJECT_ROOT, "code", "02_networks", "00.modified_sparCC.R"))

# Load previously filtered IBD data
#------------------------------------------------------------------
IBD_composite <- readRDS(file.path(DATA_ABUNDANCE, "IBD_studies_counts_filtered_composite.rds"))

# HMP will be used for discovery and HallAB for validation, so I separate them here
HMP_composite <- IBD_composite[["HMP_2019_ibdmdb"]]
Hall_2017_composite <- IBD_composite[["HallAB_2017"]]

# Separate metadata and taxa
metadata_cols <- names(HMP_composite)[!startsWith(names(HMP_composite), "k__")]
taxa_cols <- names(HMP_composite)[startsWith(names(HMP_composite), "k__")]

metadata_cols_Hall <- names(Hall_2017_composite)[!startsWith(names(Hall_2017_composite), "k__")]
taxa_cols_Hall <- names(Hall_2017_composite)[startsWith(names(Hall_2017_composite), "k__")]

HMP_composite_per_subject <- HMP_composite %>%
    group_by(subject_id) %>%
    group_split()

Hall_composite_per_subject <- Hall_2017_composite %>%
    group_by(subject_id) %>%
    group_split()

# Put subject names as list names
names(HMP_composite_per_subject) <- map_chr(HMP_composite_per_subject, ~ unique(.x$subject_id))
names(Hall_composite_per_subject) <- map_chr(Hall_composite_per_subject, ~ unique(.x$subject_id))

# Remove duplicated timepoints and subject with less than 5 timepoints
HMP_composite_per_subject <- HMP_composite_per_subject %>%
    map(~ .x %>%
        filter(!duplicated(days_from_first_collection))) %>%
    keep(~ nrow(.) >= 7) # at least 7 timepoints

Hall_composite_per_subject <- Hall_composite_per_subject %>%
    map(~ .x %>%
        filter(!duplicated(days_from_first_collection))) %>%
    keep(~ nrow(.) >= 7) # at least 7 timepoints


# Separate count tables and metadata tables per subject for HMP
#------------------------------------------------------------------
HMP_counts_per_subject <- map(HMP_composite_per_subject, ~ .x %>%
    # arrange by days from first collection
    arrange(days_from_first_collection) %>%
    select(all_of(taxa_cols), id) %>%
    column_to_rownames("id") %>%
    # Remove cols that sum to 0 (absent taxa in all samples)
    select(where(~ sum(.) > 0)))

HMP_metadata_per_subject <- map(HMP_composite_per_subject, ~ .x %>%
    arrange(days_from_first_collection) %>%
    select(all_of(metadata_cols)) %>%
    column_to_rownames("id"))

# Separate count tables and metadata tables per subject for HallAB
#------------------------------------------------------------------
Hall_counts_per_subject <- map(Hall_composite_per_subject, ~ .x %>%
    # arrange by days from first collection
    arrange(days_from_first_collection) %>%
    select(all_of(taxa_cols_Hall), id) %>%
    column_to_rownames("id") %>%
    # Remove cols that sum to 0 (absent taxa in all samples)
    select(where(~ sum(.) > 0)))

Hall_metadata_per_subject <- map(Hall_composite_per_subject, ~ .x %>%
    arrange(days_from_first_collection) %>%
    select(all_of(metadata_cols_Hall)) %>%
    column_to_rownames("id"))

# Infer SparCC networks per subject
#------------------------------------------------------------------
# Use modified SparCC function from code/02_networks/00.modified_sparCC.R (it includes a detrending step in the CLR space)
asso_thresholds <- seq(0, 0.9, by = 0.1) # thresholds for filtering associations to create adjacency matrices

# Infer SparCC networks and association matrices for HMP (discovery cohort)
withr::with_seed(
    seed = 1234,
    code = {
        subject_networks_species_sparcc <- lapply(HMP_counts_per_subject, function(counts) {
            sparcc.modified(as.matrix(counts),
                iter = 100, inner_iter = 30, detrend = TRUE
            )
        })

        subject_assomatrices_species_sparcc <- lapply(subject_networks_species_sparcc, function(network) {
            asso <- network$Cor
        })

        subject_assomatrices_species_sparcc <- map2(subject_assomatrices_species_sparcc, names(subject_assomatrices_species_sparcc), ~ {
            mat <- .x
            colnames(mat) <- rownames(mat) <- colnames(HMP_counts_per_subject[[.y]])
            return(mat)
        })
    }
)

# Infer SparCC networks and association matrices for HallAB (validation cohort)
withr::with_seed(
    seed = 1234,
    code = {
        subject_networks_species_sparcc_Hall <- lapply(Hall_counts_per_subject, function(counts) {
            sparcc.modified(as.matrix(counts),
                iter = 100, inner_iter = 30, detrend = TRUE
            )
        })

        subject_assomatrices_species_sparcc_Hall <- lapply(subject_networks_species_sparcc_Hall, function(network) {
            asso <- network$Cor
        })

        subject_assomatrices_species_sparcc_Hall <- map2(subject_assomatrices_species_sparcc_Hall, names(subject_assomatrices_species_sparcc_Hall), ~ {
            mat <- .x
            colnames(mat) <- rownames(mat) <- colnames(Hall_counts_per_subject[[.y]])
            return(mat)
        })
    }
)

# save files
#------------------------------------------------------------------
save_path <- DATA_ASSOC_MATRICES
dir.create(save_path, recursive = TRUE, showWarnings = FALSE)
saveRDS(subject_assomatrices_species_sparcc, file.path(save_path, "subject_assomatrices_species_sparcc_HMP.rds")) # raw association matrices
saveRDS(subject_assomatrices_species_sparcc_Hall, file.path(save_path, "subject_assomatrices_species_sparcc_Hall.rds")) # raw association matrices

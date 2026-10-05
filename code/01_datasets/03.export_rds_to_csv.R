# Locate the project root (from any working directory inside the repo) and load shared paths
.root <- getwd(); while (!file.exists(file.path(.root, "code", "common", "paths.R")) && dirname(.root) != .root) .root <- dirname(.root)
source(file.path(.root, "code", "common", "paths.R"))

library(tidyverse)

# Load previously processed RDS files.
#--------------------------------------------------
IBD_composite <- readRDS(file.path(DATA_ABUNDANCE, "IBD_studies_counts_filtered_composite.rds"))
IBD_composite_baseline <- readRDS(file.path(DATA_ABUNDANCE, "IBD_studies_counts_filtered_baseline_composite.rds"))

# Get HMP data
HMP_composite <- IBD_composite[["HMP_2019_ibdmdb"]]
HMP_composite_baseline <- IBD_composite_baseline[["HMP_2019_ibdmdb"]]

# Get Hall data
Hall_composite <- IBD_composite[["HallAB_2017"]]
Hall_composite_baseline <- IBD_composite_baseline[["HallAB_2017"]]

# Save as CSV files
save_path <- DATA_ABUNDANCE
dir.create(save_path, showWarnings = FALSE, recursive = TRUE)
write_csv(HMP_composite, file.path(save_path, "HMP_counts_filtered_composite.csv"))
write_csv(HMP_composite_baseline, file.path(save_path, "HMP_counts_filtered_baseline_composite.csv"))
write_csv(Hall_composite, file.path(save_path, "Hall_counts_filtered_composite.csv"))
write_csv(Hall_composite_baseline, file.path(save_path, "Hall_counts_filtered_baseline_composite.csv"))

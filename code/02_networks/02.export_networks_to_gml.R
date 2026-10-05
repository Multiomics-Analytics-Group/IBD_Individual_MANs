library(tidyverse)
library(igraph)

# Load previously inferred subject networks at species level.
#---------------------------------------------------------------
subject_assomatrices_species_sparcc <- readRDS("data/processed/IBD_CuratedMetagenomicData/subject_networks/timepoints/subject_assomatrices_species_sparcc_HMP.rds")
subject_assomatrices_species_sparcc_Hall <- readRDS("data/processed/IBD_CuratedMetagenomicData/subject_networks/timepoints/subject_assomatrices_species_sparcc_Hall.rds")

# Save individual networks as GML files after individual network formatting
#---------------------------------------------------------------
create_subject_graphs <- function(subject_adjamatrices) {
    lapply(subject_adjamatrices, function(adj_matrix) {
        diag(adj_matrix) <- 0 # Remove self edges
        adj_matrix <- adj_matrix # Ensure original signed values
        g <- graph_from_adjacency_matrix(adj_matrix, mode = "undirected", diag = FALSE, weighted = TRUE)
        V(g)$label <- V(g)$name # Set vertex labels to node names
        return(g)
    })
}

# Make graphs from association matrices
subject_graphs_species_sparcc_assoc <- create_subject_graphs(subject_assomatrices_species_sparcc)
subject_graphs_species_sparcc_assoc_Hall <- create_subject_graphs(subject_assomatrices_species_sparcc_Hall)

common_path <- "data/processed/IBD_CuratedMetagenomicData/subject_networks/gml/"
timepoints_common <- file.path(common_path, "timepoints/")

# TIMEPOINTS
# Discovery cohort (HMP)
dir.create(paste0(timepoints_common, "HMP/"), showWarnings = FALSE, recursive = TRUE)
lapply(names(subject_graphs_species_sparcc_assoc), function(subject_id) {
    g <- subject_graphs_species_sparcc_assoc[[subject_id]]
    write_graph(g, file.path(timepoints_common, "HMP/", paste0(subject_id, ".gml")), format = "gml")
})

# Validation cohort (HallAB)
dir.create(paste0(timepoints_common, "Hall/"), showWarnings = FALSE, recursive = TRUE)
lapply(names(subject_graphs_species_sparcc_assoc_Hall), function(subject_id) {
    g <- subject_graphs_species_sparcc_assoc_Hall[[subject_id]]
    write_graph(g, file.path(timepoints_common, "Hall/", paste0(subject_id, ".gml")), format = "gml")
})

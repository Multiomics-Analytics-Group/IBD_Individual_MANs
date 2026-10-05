# Central definition of every input/output location used by the R scripts.
# Keep in sync with paths.py. To relocate data or results, edit only these two files.
#
# Each script loads this file with the two bootstrap lines at its top, which work
# both with `Rscript` and interactively, from any working directory inside the repo.

PROJECT_ROOT <- local({
  root <- getwd()
  while (!file.exists(file.path(root, "code", "common", "paths.R")) && dirname(root) != root) root <- dirname(root)
  root
})

# --- Data ---------------------------------------------------------------------
DATA_RAW             <- file.path(PROJECT_ROOT, "data", "raw", "IBD_CuratedMetagenomicData")        # counts/metadata as downloaded
DATA_EXTERNAL        <- file.path(PROJECT_ROOT, "data", "external")                                 # third-party files (FUGAsseM, HMP2 metadata, dysbiosis scores)
DATA_ABUNDANCE       <- file.path(PROJECT_ROOT, "data", "processed", "IBD_CuratedMetagenomicData")  # filtered abundance tables
DATA_ASSOC_MATRICES  <- file.path(DATA_ABUNDANCE, "subject_networks", "timepoints")                 # per-subject SparCC association matrices
DATA_NETWORKS_GML    <- file.path(DATA_ABUNDANCE, "subject_networks", "gml", "timepoints")          # unthresholded networks: HMP/, Hall/
DATA_FUNCTIONAL_ATTR <- file.path(PROJECT_ROOT, "data", "external", "functional_attributes")        # GO-term node features: HMP/
DATA_PYG             <- file.path(PROJECT_ROOT, "data", "processed", "pyg_datasets")                # PyG datasets and thresholded GML

# --- Results: models ----------------------------------------------------------
RES_TRAD_PICKLES <- file.path(PROJECT_ROOT, "results", "machine_learning", "traditional_models")
RES_TRAD_TABLES  <- file.path(RES_TRAD_PICKLES, "converted_pickles")
RES_GNN_PICKLES  <- file.path(PROJECT_ROOT, "results", "machine_learning", "graph_based_models",
                              "HMP_w_functions_onehot_graphtypes", "timepoints_self")
RES_GNN_TABLES   <- file.path(RES_GNN_PICKLES, "converted_pickles")

# --- Results: interpretation (integrated gradients) ---------------------------
RES_INTERP_SEEDS  <- file.path(PROJECT_ROOT, "results", "machine_learning", "graph_based_models",
                               "model_interpretation", "HMP_timepoints_self_t_0_3", "interpretation_10_seeds")  # seed_*/ folders
RES_INTERP_AGG    <- RES_INTERP_SEEDS                                                                # all_seeds_*.csv
RES_INTERP_TABLES <- file.path(PROJECT_ROOT, "results", "tables", "analyses", "attributions")        # modules, top-20 features

# --- Results: ablation --------------------------------------------------------
RES_ABLATION_GNN     <- PROJECT_ROOT                                                                 # cross_evaluation_*.csv, e_coli_graph_metrics.csv
RES_ABLATION_NETFEAT <- file.path(PROJECT_ROOT, "results", "tables", "analyses", "attributions")     # per-subject network features
RES_ABLATION_RF      <- file.path(PROJECT_ROOT, "results", "machine_learning", "gnn_top20_graphnodes")

# --- Figures (one folder per producing script) --------------------------------
FIG_DIVERSITY       <- file.path(PROJECT_ROOT, "results", "plots", "analyses", "diversity")
FIG_NETWORK_METRICS <- file.path(PROJECT_ROOT, "results", "plots", "analyses", "network_metrics")
FIG_CLASSIFICATION  <- file.path(PROJECT_ROOT, "results", "plots", "machine_learning")
FIG_INTERPRETATION  <- file.path(PROJECT_ROOT, "results", "plots", "analyses", "attributions")
FIG_RF_FEATURES     <- file.path(PROJECT_ROOT, "results", "plots", "analyses", "attributions")
FIG_MODULES         <- file.path(PROJECT_ROOT, "results", "plots", "analyses", "modules")
FIG_ECOLI           <- file.path(PROJECT_ROOT, "results", "plots", "analyses", "ecoli")

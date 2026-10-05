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

# --- Data ----------------------------------------------------------------------
DATA_RAW             <- file.path(PROJECT_ROOT, "data", "raw", "curatedMetagenomicData")                  # counts/metadata as downloaded from curatedMetagenomicData
DATA_EXTERNAL        <- file.path(PROJECT_ROOT, "data", "external")                                       # third-party files (FUGAsseM, HMP2 metadata, dysbiosis scores)
DATA_ABUNDANCE       <- file.path(PROJECT_ROOT, "data", "processed", "abundance")                         # filtered abundance tables (.rds, .csv)
DATA_ASSOC_MATRICES  <- file.path(PROJECT_ROOT, "data", "processed", "networks", "association_matrices")  # per-subject SparCC association matrices
DATA_NETWORKS_GML    <- file.path(PROJECT_ROOT, "data", "processed", "networks", "gml")                   # unthresholded per-subject networks: HMP/, Hall/
DATA_FUNCTIONAL_ATTR <- file.path(PROJECT_ROOT, "data", "processed", "functional_attributes")             # GO-term node features: HMP/
DATA_PYG             <- file.path(PROJECT_ROOT, "data", "processed", "pyg_datasets")                      # PyG datasets, CLR/taxonomy tables and thresholded GML

# --- Results: models -----------------------------------------------------------
RES_TRAD_PICKLES     <- file.path(PROJECT_ROOT, "results", "models", "traditional", "pickles")  # raw output of 03_train_traditional_models.py
RES_TRAD_TABLES      <- file.path(PROJECT_ROOT, "results", "models", "traditional", "tables")   # tidy CSVs made by 05_export_model_results_csv.py
RES_GNN_PICKLES      <- file.path(PROJECT_ROOT, "results", "models", "gnn", "pickles")          # raw output of 04_train_gnn_models.py
RES_GNN_TABLES       <- file.path(PROJECT_ROOT, "results", "models", "gnn", "tables")           # tidy CSVs made by 05_export_model_results_csv.py

# --- Results: interpretation (integrated gradients) ----------------------------
RES_INTERP_SEEDS     <- file.path(PROJECT_ROOT, "results", "interpretation", "per_seed")  # seed_*/ folders of integrated-gradients attributions
RES_INTERP_AGG       <- file.path(PROJECT_ROOT, "results", "interpretation")              # all_seeds_*.csv aggregated over seeds
RES_INTERP_TABLES    <- file.path(PROJECT_ROOT, "results", "interpretation", "tables")    # module membership, top-20 features

# --- Results: ablation ---------------------------------------------------------
RES_ABLATION_GNN     <- file.path(PROJECT_ROOT, "results", "ablation", "gnn")               # cross_evaluation_*_modules.csv (GNN module ablation)

# --- Figures (one folder per producing script) ---------------------------------
FIG_DIVERSITY        <- file.path(PROJECT_ROOT, "results", "figures", "diversity")
FIG_NETWORK_METRICS  <- file.path(PROJECT_ROOT, "results", "figures", "network_metrics")
FIG_CLASSIFICATION   <- file.path(PROJECT_ROOT, "results", "figures", "classification")
FIG_INTERPRETATION   <- file.path(PROJECT_ROOT, "results", "figures", "interpretation")
FIG_RF_FEATURES      <- file.path(PROJECT_ROOT, "results", "figures", "rf_features")
FIG_MODULES          <- file.path(PROJECT_ROOT, "results", "figures", "modules")
FIG_ECOLI            <- file.path(PROJECT_ROOT, "results", "figures", "ecoli")

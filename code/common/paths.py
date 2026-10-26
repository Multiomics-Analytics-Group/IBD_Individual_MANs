"""Central definition of every input/output location used by the Python scripts.

Keep in sync with paths.R. To relocate data or results, edit only these two files.

Each script loads this module with the bootstrap lines at its top, which work both
from the terminal and interactively, from any working directory inside the repo.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Data ----------------------------------------------------------------------
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "curatedMetagenomicData"                              # counts/metadata as downloaded from curatedMetagenomicData
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"                                               # third-party files (FUGAsseM, HMP2 metadata, dysbiosis scores)
DATA_ABUNDANCE = PROJECT_ROOT / "data" / "processed" / "abundance"                               # filtered abundance tables (.rds, .csv)
DATA_ASSOC_MATRICES = PROJECT_ROOT / "data" / "processed" / "networks" / "association_matrices"  # per-subject SparCC association matrices
DATA_NETWORKS_GML = PROJECT_ROOT / "data" / "processed" / "networks" / "gml"                     # unthresholded per-subject networks: HMP/, Hall/
DATA_FUNCTIONAL_ATTR = PROJECT_ROOT / "data" / "processed" / "functional_attributes"             # GO-term node features: HMP/
DATA_PYG = PROJECT_ROOT / "data" / "processed" / "pyg_datasets"                                  # PyG datasets, CLR/taxonomy tables and thresholded GML

# --- Results: models -----------------------------------------------------------
RES_TRAD_PICKLES = PROJECT_ROOT / "results" / "models" / "traditional" / "pickles"  # raw output of 03_train_traditional_models.py
RES_TRAD_TABLES = PROJECT_ROOT / "results" / "models" / "traditional" / "tables"    # tidy CSVs made by 05_export_model_results_csv.py
RES_GNN_PICKLES = PROJECT_ROOT / "results" / "models" / "gnn" / "pickles"           # raw output of 04_train_gnn_models.py
RES_GNN_TABLES = PROJECT_ROOT / "results" / "models" / "gnn" / "tables"             # tidy CSVs made by 05_export_model_results_csv.py

# --- Results: interpretation (integrated gradients) ----------------------------
RES_INTERP_SEEDS = PROJECT_ROOT / "results" / "interpretation" / "per_seed"  # seed_*/ folders of integrated-gradients attributions
RES_INTERP_AGG = PROJECT_ROOT / "results" / "interpretation"                 # all_seeds_*.csv aggregated over seeds

# --- Results: ablation ---------------------------------------------------------
RES_ABLATION = PROJECT_ROOT / "results" / "ablation"                                              # metagraph_species.csv
RES_ABLATION_GNN = PROJECT_ROOT / "results" / "ablation" / "gnn"                                    # cross_evaluation_*_modules.csv, tables and saved models
RES_BACTEROIDALES = PROJECT_ROOT / "results" / "ablation" / "bacteroidales_associations"            # SparCC weights among metagraph Bacteroidales

# --- Figures (one folder per producing script) ---------------------------------
FIG_DIVERSITY = PROJECT_ROOT / "results" / "figures" / "diversity"
FIG_NETWORK_METRICS = PROJECT_ROOT / "results" / "figures" / "network_metrics"
FIG_CLASSIFICATION = PROJECT_ROOT / "results" / "figures" / "classification"
FIG_INTERPRETATION = PROJECT_ROOT / "results" / "figures" / "interpretation"
FIG_RF_FEATURES = PROJECT_ROOT / "results" / "figures" / "rf_features"
FIG_ABLATION = PROJECT_ROOT / "results" / "figures" / "ablation"
FIG_BACTEROIDALES = PROJECT_ROOT / "results" / "figures" / "bacteroidales_associations"

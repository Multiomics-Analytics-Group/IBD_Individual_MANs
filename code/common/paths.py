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
RES_TRAD_PICKLES = PROJECT_ROOT / "results" / "models" / "traditional" / "pickles"  # raw output of 03.traditional_models.py
RES_TRAD_TABLES = PROJECT_ROOT / "results" / "models" / "traditional" / "tables"    # tidy CSVs made by 05.convert_pickles_to_csv.py
RES_GNN_PICKLES = PROJECT_ROOT / "results" / "models" / "gnn" / "pickles"           # raw output of 04.graph_based_models.py
RES_GNN_TABLES = PROJECT_ROOT / "results" / "models" / "gnn" / "tables"             # tidy CSVs made by 05.convert_pickles_to_csv.py

# --- Results: interpretation (integrated gradients) ----------------------------
RES_INTERP_SEEDS = PROJECT_ROOT / "results" / "interpretation" / "per_seed"  # seed_*/ folders of integrated-gradients attributions
RES_INTERP_AGG = PROJECT_ROOT / "results" / "interpretation"                 # all_seeds_*.csv aggregated over seeds
RES_INTERP_TABLES = PROJECT_ROOT / "results" / "interpretation" / "tables"   # module membership, top-20 features

# --- Results: ablation ---------------------------------------------------------
RES_ABLATION_GNN = PROJECT_ROOT / "results" / "ablation" / "gnn"                   # cross_evaluation_*.csv, e_coli_graph_metrics.csv
RES_ABLATION_NETFEAT = PROJECT_ROOT / "results" / "ablation" / "network_features"  # per-subject network features used by the RF ablation
RES_ABLATION_RF = PROJECT_ROOT / "results" / "ablation" / "rf"                     # output folder of 03_RF_ablation.py

# --- Figures (one folder per producing script) ---------------------------------
FIG_DIVERSITY = PROJECT_ROOT / "results" / "figures" / "diversity"
FIG_NETWORK_METRICS = PROJECT_ROOT / "results" / "figures" / "network_metrics"
FIG_CLASSIFICATION = PROJECT_ROOT / "results" / "figures" / "classification"
FIG_INTERPRETATION = PROJECT_ROOT / "results" / "figures" / "interpretation"
FIG_RF_FEATURES = PROJECT_ROOT / "results" / "figures" / "rf_features"
FIG_MODULES = PROJECT_ROOT / "results" / "figures" / "modules"
FIG_ECOLI = PROJECT_ROOT / "results" / "figures" / "ecoli"

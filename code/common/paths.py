"""Central definition of every input/output location used by the Python scripts.

Keep in sync with paths.R. To relocate data or results, edit only these two files.

Each script loads this module with the bootstrap lines at its top, which work both
from the terminal and interactively, from any working directory inside the repo.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Data ---------------------------------------------------------------------
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "IBD_CuratedMetagenomicData"              # counts/metadata as downloaded
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"                                   # third-party files (FUGAsseM, HMP2 metadata, dysbiosis scores)
DATA_ABUNDANCE = PROJECT_ROOT / "data" / "processed" / "IBD_CuratedMetagenomicData"  # filtered abundance tables
DATA_ASSOC_MATRICES = DATA_ABUNDANCE / "subject_networks" / "timepoints"             # per-subject SparCC association matrices
DATA_NETWORKS_GML = DATA_ABUNDANCE / "subject_networks" / "gml" / "timepoints"       # unthresholded networks: HMP/, Hall/
DATA_FUNCTIONAL_ATTR = PROJECT_ROOT / "data" / "external" / "functional_attributes"  # GO-term node features: HMP/
DATA_PYG = PROJECT_ROOT / "data" / "processed" / "pyg_datasets"                      # PyG datasets and thresholded GML

# --- Results: models ----------------------------------------------------------
RES_TRAD_PICKLES = PROJECT_ROOT / "results" / "machine_learning" / "traditional_models"
RES_TRAD_TABLES = RES_TRAD_PICKLES / "converted_pickles"
RES_GNN_PICKLES = (PROJECT_ROOT / "results" / "machine_learning" / "graph_based_models"
                   / "HMP_w_functions_onehot_graphtypes" / "timepoints_self")
RES_GNN_TABLES = RES_GNN_PICKLES / "converted_pickles"

# --- Results: interpretation (integrated gradients) ---------------------------
RES_INTERP_SEEDS = (PROJECT_ROOT / "results" / "machine_learning" / "graph_based_models"
                    / "model_interpretation" / "HMP_timepoints_self_t_0_3" / "interpretation_10_seeds")  # seed_*/ folders
RES_INTERP_AGG = RES_INTERP_SEEDS                                                     # all_seeds_*.csv
RES_INTERP_TABLES = PROJECT_ROOT / "results" / "tables" / "analyses" / "attributions"  # modules, top-20 features

# --- Results: ablation --------------------------------------------------------
RES_ABLATION_GNN = PROJECT_ROOT                                                        # cross_evaluation_*.csv, e_coli_graph_metrics.csv
RES_ABLATION_NETFEAT = PROJECT_ROOT / "results" / "tables" / "analyses" / "attributions"  # per-subject network features
RES_ABLATION_RF = PROJECT_ROOT / "results" / "machine_learning" / "gnn_top20_graphnodes"

# --- Figures (one folder per producing script) --------------------------------
FIG_DIVERSITY = PROJECT_ROOT / "results" / "plots" / "analyses" / "diversity"
FIG_NETWORK_METRICS = PROJECT_ROOT / "results" / "plots" / "analyses" / "network_metrics"
FIG_CLASSIFICATION = PROJECT_ROOT / "results" / "plots" / "machine_learning"
FIG_INTERPRETATION = PROJECT_ROOT / "results" / "plots" / "analyses" / "attributions"
FIG_RF_FEATURES = PROJECT_ROOT / "results" / "plots" / "analyses" / "attributions"
FIG_MODULES = PROJECT_ROOT / "results" / "plots" / "analyses" / "modules"
FIG_ECOLI = PROJECT_ROOT / "results" / "plots" / "analyses" / "ecoli"

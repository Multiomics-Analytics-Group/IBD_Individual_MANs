import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import networkx as nx
from torch_geometric.utils import add_self_loops


# --- SETUP & CONFIGURATION ---
# Locate the project root (works from the terminal and interactively) and load shared paths
_start = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
_root = next(p for p in (_start, *_start.parents) if (p / "code" / "common" / "paths.py").exists())
sys.path.insert(0, str(_root / "code" / "common"))
import paths

# Loading custom functions from code/common/helper_functions.py
import helper_functions as hf

PROCESSED_DATA_PATH = paths.DATA_ABUNDANCE
RESULTS_PATH = paths.DATA_PYG
FUNCTIONS_PATH = paths.DATA_FUNCTIONAL_ATTR / "HMP"


# Define the dataset metadata
DATASETS = {
    "HMP": {
        "is_train": True,
        "meta_file": "HMP_counts_filtered_baseline_composite.csv",
        "composite_file": "HMP_counts_filtered_composite.csv",
        "binary_map": {"healthy": 1, "IBD": 0}
    },
    "Hall": {
        "is_train": False,
        "meta_file": "Hall_counts_filtered_baseline_composite.csv",
        "composite_file": "Hall_counts_filtered_composite.csv",
        "binary_map": {"healthy": 1, "IBD": 0}
    }
}

# Define Network Types and their paths
NET_CONFIGS = {
    "timepoints": paths.DATA_NETWORKS_GML
}

# Define thresholds and taxonomic ranks for filtering
THRESHOLDS = [0.5, 0.4, 0.3]
TAXONOMIC_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

# --- REUSABLE HELPER FUNCTIONS ---
def filter_nets(nets_dict, thresholds, ref_net=None, weight_key="weight"):
    out = {}

    for t in thresholds:
        t_str = str(t)
        filtered = {}

        for nid, graph in nets_dict.items():
            G = graph.copy()

            to_rem = []
            for u, v in G.edges():
                if ref_net is not None:

                    # Get weight from reference; if edge doesn't exist in ref, weight = 0
                    ref_edge_data = ref_net.get_edge_data(u, v, default={})
                    val = abs(ref_edge_data.get(weight_key, 0))

                else:
                    # Standard thresholding on the graph itself
                    val = abs(G[u][v].get(weight_key, 0))
                
                if val < t:
                    to_rem.append((u, v))

            G.remove_edges_from(to_rem)

            filtered[nid] = G

            # print(f"t: {t}, Subject {nid}: Remaining edges: {G.number_of_edges()}, Removed edges: {len(to_rem)}")
            # print(f"t: {t}, Subject {nid}: Remaining nodes: {G.number_of_nodes()}")

        out[t_str] = filtered

    return out


def drop_unclassified(networks_by_thresh, rank="s__"):
    for t_str, nets in networks_by_thresh.items():
        for nid, G in nets.items():
            to_rem = [n for n in G.nodes if any(s.startswith(rank) and 'unclassified' in s[len(rank):].lower() for s in n.split('|'))]
            G.remove_nodes_from(to_rem)
    return networks_by_thresh


# --- EXECUTION PIPELINE ---

# Load Raw Networks
raw_networks = {ds: {nt: hf.load_networks(path / ds) for nt, path in NET_CONFIGS.items()} for ds in DATASETS}
filtered_networks = {
    "HMP": {}, "Hall": {}
}

# Sparsification
for ds in DATASETS:

    # Standard thresholding
    filtered_networks[ds]["timepoints_self"] = filter_nets(raw_networks[ds]["timepoints"], THRESHOLDS)


# Drop unclassified species
for ds in DATASETS:
    for nt in filtered_networks[ds].keys():
        filtered_networks[ds][nt] = drop_unclassified(filtered_networks[ds][nt])

# Load Metadata & Labels
labels = {}
abundance_data = {}

for ds, cfg in DATASETS.items():
    meta = pd.read_csv(PROCESSED_DATA_PATH / cfg["meta_file"], index_col="subject_id")
    y_multi = meta["condition"]
    multi_map = {cond: i for i, cond in enumerate(y_multi.unique())}
    labels[ds] = {
        "binary": y_multi.replace({"CD": "IBD", "UC": "IBD"}).map(cfg["binary_map"]),
        "multi": y_multi.map(multi_map)
    }
    
    comp = pd.read_csv(PROCESSED_DATA_PATH / cfg["composite_file"], index_col="id")
    abundance_data[ds] = {
        "counts": comp.loc[:, comp.columns.str.startswith("k__")],
        "subject_ids": comp["subject_id"]
    }

# Load functions
func_emb_dict = {
    "emb_BP_idf": pd.read_csv(FUNCTIONS_PATH / "taxa_one_hot_encoded_one_hot_bp_weighted.csv", index_col=0),
    "emb_MF_idf": pd.read_csv(FUNCTIONS_PATH / "taxa_one_hot_encoded_one_hot_mf_weighted.csv", index_col=0)
}

# --- FEATURE MANIPULATION AND PYG CONVERSION ---

# Prepare to remove subjects with fewer than 7 timepoints.
valid_subjects_by_ds = {}
for ds in DATASETS:
    counts = abundance_data[ds]["subject_ids"].value_counts()
    valid_subjects_by_ds[ds] = counts[counts >= 7].index.tolist()
    print(f"{ds}: Found {len(valid_subjects_by_ds[ds])} subjects with >= 7 timepoints.")

# Gather all unique threshold strings present across all network types
all_t_strs = set()
for ds in DATASETS:
    for nt in filtered_networks[ds].keys():
        all_t_strs.update(filtered_networks[ds][nt].keys())

# Iterate through all discovered thresholds (e.g., '0.5', '0.2', etc.)
for t_str in sorted(list(all_t_strs)):
    
    # build a reference hierarchy from HMP for this threshold to use in Hall
    hmp_hierarchies = {} 

    for ds in DATASETS:
        for nt in filtered_networks[ds].keys():
            
            # Skip if this specific network type doesn't have this threshold 
            if t_str not in filtered_networks[ds][nt]:
                continue
                
            all_nets = filtered_networks[ds][nt][t_str]
            
            # Filter networks to only include those from valid subjects (those with >= 7 timepoints)
            nets = {nid: G for nid, G in all_nets.items() if nid in valid_subjects_by_ds[ds]}
            
            # Skip if no networks remain for this subject/threshold combo
            if not nets:
                continue
                
            print(f"{ds} - {nt} - Threshold {t_str}: Retained {len(nets)} graphs after filtering for valid subjects.")
            
            # Get unique taxa in this network set
            all_taxa = {node for G in nets.values() for node in G.nodes}
            
            # build Hierarchy
            taxa_df = pd.DataFrame([hf.parse_taxon(tx, TAXONOMIC_RANKS) for tx in all_taxa], index=list(all_taxa))
            taxa_df = taxa_df.loc[:, taxa_df.nunique() > 1].drop(columns=["species"], errors='ignore')
            hierarchy = pd.get_dummies(taxa_df, columns=taxa_df.columns, prefix='', prefix_sep='')
            
            if ds == "HMP":
                hmp_hierarchies[nt] = hierarchy # Store for Hall ref
            
            # Compute CLR, aggregate by subject, and calculate mean/variance for each taxon
            taxa_cols = [c for c in abundance_data[ds]["counts"].columns if c in all_taxa]
            clr_vals = hf.clr(abundance_data[ds]["counts"][taxa_cols])

            mean_clr = clr_vals.groupby(abundance_data[ds]["subject_ids"]).mean().T
            var_clr = clr_vals.groupby(abundance_data[ds]["subject_ids"]).var().T
            
            # Add features
            hf.add_node_features(
                nets, 
                hmp_hierarchies[nt], 
                hierarchy if ds != "HMP" else None,
                mean_clr, var_clr, func_emb_dict,
                is_training_dataset=DATASETS[ds]["is_train"]
            )

            # Create PyG Dataset
            pyg_dict = hf.create_pyg_dataset(nets, labels[ds]["binary"], labels[ds]["multi"])
            dataset = hf.GraphDictDataset(list(pyg_dict.values()))
            
            # Save
            out_dir = RESULTS_PATH / f"{ds}_{nt}_w_functions_onehot"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Clean filename string
            suffix = t_str.replace('.', '_')
            save_path = out_dir / f"pyg_dataset_species_threshold_assoc_{suffix}.pt"
            save_path_clr = out_dir / f"clr_values_species_threshold_assoc_{suffix}.csv"
            save_path_var = out_dir / f"clr_variance_species_threshold_assoc_{suffix}.csv"
            save_path_hierarchy = out_dir / f"hierarchy_species_threshold_assoc_{suffix}.csv"
            
            torch.save(dataset, save_path)
            pd.DataFrame(mean_clr).to_csv(save_path_clr)
            pd.DataFrame(var_clr).to_csv(save_path_var)
            pd.DataFrame(hierarchy).to_csv(save_path_hierarchy)
            print(f"Saved: {save_path}")
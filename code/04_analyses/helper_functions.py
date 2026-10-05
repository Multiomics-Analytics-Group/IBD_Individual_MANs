from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
from scipy.stats import gmean
import numpy as np
import networkx as nx
import pandas as pd
import torch

# Class to convert NetworkX graphs to PyTorch Geometric Data objects


class GraphDictDataset(InMemoryDataset):

    def __init__(self, graphs_list):
        super().__init__('.', transform=None, pre_transform=None)
        self.data, self.slices = self.collate(graphs_list)

    def get(self, idx):
        return super().get(idx)

# Function to load networks into nested dictionaries


def load_networks(path):
    networks = {}
    for gml_file in path.glob("*.gml"):
        subject_name = gml_file.stem
        networks[subject_name] = nx.read_gml(gml_file)
    return networks

# Function to parse taxonomic strings into DataFrame rows


def parse_taxon(taxon, ranks):
    parts = taxon.split('|')
    values = []

    for i, rank in enumerate(ranks):
        # Extract the corresponding part if it exists
        if i < len(parts) and "__" in parts[i]:
            val = parts[i].split("__", 1)[1].strip()
        else:
            val = ""

        # Handle unclassified or missing taxa
        if (not val) or ("unclassified" in val.lower()) or ("unknown" in val.lower()):
            val = f"{val}_{rank}" if val else f"unknown_{rank}"

        values.append(val)

    # Pad missing levels with rank-specific unknowns
    while len(values) < len(ranks):
        rank = ranks[len(values)]
        values.append(f"unknown_{rank}")

    # Disambiguate repeated names across ranks (such as "Actinobacteria" in phylum and class)
    seen = {}
    for j, v in enumerate(values):
        if v in seen:
            # Same name appeared in an upper rank — disambiguate
            values[j] = f"{v}_{ranks[j]}"
        seen[v] = True

    return pd.Series(values[:len(ranks)], index=ranks)

# Function to transform counts to clr


def clr(df, pseudocount=1):
    df = df + pseudocount  # avoid log(0)
    gm = gmean(df, axis=1)
    clr_df = np.log(df.div(gm, axis=0))
    return clr_df

def relabundance(df):
    relab_df = df.div(df.sum(axis=1), axis=0)
    return relab_df

# Function to add node features to graphs in the network dictionary
# def add_node_features(network_dict, taxa_hierarchy, taxa_hierarchy_emb64, taxa_hierarchy_emb128, subject_abundance):

#     num_hierarchy_features = taxa_hierarchy.shape[1]
#     num_emb64_features = taxa_hierarchy_emb64.shape[1]
#     num_emb128_features = taxa_hierarchy_emb128.shape[1]

#     for subject, G in network_dict.items():

#         for node in G.nodes:
#             G.nodes[node]["node_name"] = node  # store original name

#             node_exists = all([
#                 node in taxa_hierarchy.index,
#                 node in taxa_hierarchy_emb64.index,
#                 node in taxa_hierarchy_emb128.index,
#                 node in subject_abundance.index
#             ])

#             if node_exists:
#                 # Flatten features to 1D arrays
#                 hierarchy_features = np.array(
#                     taxa_hierarchy.loc[node], dtype=float).ravel()
#                 emb64_features = np.array(
#                     taxa_hierarchy_emb64.loc[node], dtype=float).ravel()
#                 emb128_features = np.array(
#                     taxa_hierarchy_emb128.loc[node], dtype=float).ravel()
#                 abundance_feature = float(
#                     subject_abundance.loc[node, subject]) if subject in subject_abundance.columns else np.nan

#                 # Assign individual features
#                 G.nodes[node]["x_taxonomy_onehot"] = hierarchy_features.tolist()
#                 G.nodes[node]["x_taxonomy_emb_64d"] = emb64_features.tolist()
#                 G.nodes[node]["x_taxonomy_emb_128d"] = emb128_features.tolist()
#                 G.nodes[node]["x_mean_abundance"] = [abundance_feature]
#                 G.nodes[node]["x_dummy"] = [1.0]  # dummy feature

#                 # Assign combined features
#                 G.nodes[node]['x_combined_onehot'] = np.concatenate(
#                     [hierarchy_features, [abundance_feature]]).tolist()
#                 G.nodes[node]['x_combined_emb_64d'] = np.concatenate(
#                     [emb64_features, [abundance_feature]]).tolist()
#                 G.nodes[node]['x_combined_emb_128d'] = np.concatenate(
#                     [emb128_features, [abundance_feature]]).tolist()
#             else:
#                 # Fill zeros/NaN if node missing
#                 G.nodes[node]["x_taxonomy_onehot"] = np.zeros(
#                     num_hierarchy_features, dtype=float).tolist()
#                 G.nodes[node]["x_taxonomy_emb_64d"] = np.zeros(
#                     num_emb64_features, dtype=float).tolist()
#                 G.nodes[node]["x_taxonomy_emb_128d"] = np.zeros(
#                     num_emb128_features, dtype=float).tolist()
#                 G.nodes[node]["x_mean_abundance"] = np.nan
#                 G.nodes[node]['x_combined_onehot'] = np.zeros(
#                     num_hierarchy_features + 1, dtype=float).tolist()
#                 G.nodes[node]['x_combined_emb_64d'] = np.zeros(
#                     num_emb64_features + 1, dtype=float).tolist()
#                 G.nodes[node]['x_combined_emb_128d'] = np.zeros(
#                     num_emb128_features + 1, dtype=float).tolist()


def add_node_features(network_dict, taxa_hierarchy,
                      clr_abundance, rel_abundance,
                      func_embeddings_dict):

    num_hierarchy_features = taxa_hierarchy.shape[1]

    rng = np.random.default_rng(seed=13)  # For reproducibility

    for subject, G in network_dict.items():

        for node in G.nodes:
            G.nodes[node]["node_name"] = node # store original name

            # ---------- TAXONOMY ONE-HOT ----------
            if node in taxa_hierarchy.index:
                x_onehot = taxa_hierarchy.loc[node].to_numpy(dtype=float)
                has_onehot = 1.0
            else:
                x_onehot = np.zeros(num_hierarchy_features, dtype=float)
                has_onehot = 0.0

            # ---------- CLR ABUNDANCE ----------
            if subject in clr_abundance.columns and node in clr_abundance.index:
                clr_value = float(clr_abundance.loc[node, subject])
                has_clr = 1.0
            else:
                clr_value = np.nan
                has_clr = 0.0

            # ---------- REL ABUNDANCE ----------
            if subject in rel_abundance.columns and node in rel_abundance.index:
                relabund_value = float(rel_abundance.loc[node, subject])
                has_relabund = 1.0
            else:
                relabund_value = np.nan
                has_relabund = 0.0

            # ---------- FUNCTIONAL EMBEDDINGS ----------
            for emb_name, emb_table in func_embeddings_dict.items():

                num_features = emb_table.shape[1]

                if node in emb_table.index:
                    x_func = emb_table.loc[node].to_numpy(dtype=float)
                    has_func = 1.0

                else:
                    # all-0es if missing
                    x_func = np.zeros(num_features, dtype=float)
                    has_func = 0.0

                G.nodes[node][f"x_func_{emb_name}"] = x_func.tolist()
                G.nodes[node][f"has_func_{emb_name}"] = [has_func]


            # ---------- ASSIGN FEATURES ----------
            G.nodes[node]["x_taxonomy_onehot"] = x_onehot.tolist()
            G.nodes[node]["x_mean_clr"] = [clr_value]
            G.nodes[node]["x_mean_relabund"] = [relabund_value]

            # Reliability flags
            G.nodes[node]["has_onehot"] = [has_onehot]
            G.nodes[node]["has_clr"] = [has_clr]
            G.nodes[node]["has_relabund"] = [has_relabund]

            # Dummy
            G.nodes[node]["x_dummy"] = [1.0]


# Function to create PyG dataset from NetworkX graphs
def create_pyg_dataset(graphs, y_binary, y_multi):
    pyg_graphs = {}
    for graph_name, graph in graphs.items():
        data = from_networkx(graph)
        data.name = graph_name
        data.y_binary = y_binary.loc[graph_name]
        data.y_multi = y_multi.loc[graph_name]
        pyg_graphs[graph_name] = data
    return pyg_graphs

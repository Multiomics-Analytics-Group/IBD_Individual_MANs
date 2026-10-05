from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
from scipy.stats import gmean
import numpy as np
import networkx as nx
import pandas as pd

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

# Function to transform counts to clr values, with pseudocount to avoid log(0)
def clr(df, pseudocount=1):
    df = df + pseudocount  # avoid log(0)
    gm = gmean(df, axis=1)
    clr_df = np.log(df.div(gm, axis=0))
    return clr_df

# Function to add node features to graphs in the network dictionary
def add_node_features(network_dict, taxa_hierarchy_table, taxa_hierarchy_table_validation,
                      clr_abundance_table, clr_var_table, functions_dict, is_training_dataset=True):


    train_taxonomy_cols = taxa_hierarchy_table.columns

    if is_training_dataset:
        taxa_hierarchy_table_validation = None  # not used for training dataset

    for subject, G in network_dict.items():
        for node in G.nodes:
            G.nodes[node]["node_name"] = node # store original name

            # ---------- TAXONOMY ONE-HOT ----------
            if is_training_dataset:
                if node in taxa_hierarchy_table.index:
                    x_onehot = taxa_hierarchy_table.loc[node].to_numpy(dtype=float)
                    has_onehot = 1.0
                else:
                    raise ValueError(f"Node '{node}' not found in training hierarchy for subject '{subject}'")
            else:
                # For validation dataset, novel species may be present that are not in the training taxa hierarchy.
                # In this case, taxonomy information from other ranks can still be used if available in the
                # training taxa hierarchy table.
                # for instance, if Bacteroides nordii is exclusive to the validatio set, its genus and all higher ranks
                # should still be present in the training taxa hierarchy table allowing the use of that information for one-hot encoding.

                if node in taxa_hierarchy_table_validation.index:
                    if node in taxa_hierarchy_table.index:
                        # If the node is present in both training and validation taxa hierarchy tables,
                        # use the training one for consistency.
                        x_onehot = taxa_hierarchy_table.loc[node].to_numpy(dtype=float)
                        has_onehot = 1.0
                    else:
                         # If the node is only present in the validation taxa hierarchy table,
                         # use it but reindex to the training taxonomy columns to ensure consistent feature dimensions.
                        row = taxa_hierarchy_table_validation.loc[[node]].reindex(columns=train_taxonomy_cols, fill_value=0)
                        x_onehot = row.to_numpy(dtype=float).flatten()
                        has_onehot = 1.0
                else:
                    raise ValueError(f"Node '{node}' not found in taxa hierarchy for subject '{subject}'")

            # ---------- CLR ABUNDANCE ----------
            if subject in clr_abundance_table.columns and node in clr_abundance_table.index:
                clr_value = float(clr_abundance_table.loc[node, subject])
                has_clr = 1.0
            else:
                clr_value = np.nan
                has_clr = 0.0

            # ---------- CLR VARIANCE ----------
            if subject in clr_var_table.columns and node in clr_var_table.index:
                clr_var_value = float(clr_var_table.loc[node, subject])
                has_clr_var = 1.0
            else:
                clr_var_value = np.nan
                has_clr_var = 0.0

            # ---------- FUNCTIONAL VECTORS ----------
            for emb_name, emb_table in functions_dict.items():

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
            G.nodes[node]["x_clr_variance"] = [float(np.log(clr_var_value + 1e-9))]  # log-transform variance to reduce scale, add small constant to avoid log(0)

            # Reliability flags
            G.nodes[node]["has_onehot"] = [has_onehot]
            G.nodes[node]["has_clr"] = [has_clr]
            G.nodes[node]["has_clr_var"] = [has_clr_var]

            # Dummy
            G.nodes[node]["x_dummy"] = [1.0]

            # Check if clr has NaN
            if np.isnan(clr_value):
                raise(ValueError(f"CLR value for node '{node}' in subject '{subject}' is NaN. Check if the node is present in the clr abundance table."))
            
            # Check if clr variance has NaN or infinite values
            if np.isnan(clr_var_value):
                raise(ValueError(f"CLR variance value for node '{node}' in subject '{subject}' is NaN. Check if the node is present in the clr variance table."))
            if np.isinf(clr_var_value):
                raise(ValueError(f"CLR variance value for node '{node}' in subject '{subject}' is infinite. Check if the clr variance table has very large values for this node."))


# Function to create PyG dataset from NetworkX graphs
def create_pyg_dataset(graphs, y_binary_var, y_multi_var):
    pyg_graphs = {}
    for graph_name, graph in graphs.items():
        data = from_networkx(graph)
        data.name = graph_name
        data.y_binary = y_binary_var.loc[graph_name]
        data.y_multi = y_multi_var.loc[graph_name]
        pyg_graphs[graph_name] = data
    return pyg_graphs

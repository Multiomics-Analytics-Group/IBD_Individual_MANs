import argparse
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from pathlib import Path
from torch_geometric.loader import DataLoader
from captum.attr import IntegratedGradients


# Parse Command Line Arguments
# -----------------------------------------------------
parser = argparse.ArgumentParser(description="Run GNN Model Interpretation for a given seed.")
parser.add_argument("--seed", type=int, default=100, help="Random seed for reproducibility")
args, _ = parser.parse_known_args()  # tolerate the extra arguments passed by Jupyter kernels

REPRODUCIBLE_SEED = args.seed

# Locate the project root (works from the terminal and interactively) and load shared paths
_start = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
_root = next(p for p in (_start, *_start.parents) if (p / "code" / "common" / "paths.py").exists())
sys.path.insert(0, str(_root / "code" / "common"))
sys.path.insert(0, str(_root / "code" / "03_machine_learning"))  # local modules of this stage
import paths

# Custom imports
from graph_model_functions import train_model2, set_seed
from model_definitions import DualStreamGNN2

# Config
# -----------------------------------------------------
set_seed(REPRODUCIBLE_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PARAMS = {
    't_str': "0_3",
    'dataset': 'HMP', 
    'node_features_list': ['x_mean_clr', 'x_clr_variance', 'x_taxonomy_onehot', 'x_func_emb_BP_idf'],
    'pred_threshold': 0.5,
    'weight_type': 'assoc_',
    'model_type': 'GraphConv',
    'num_layers': 2,
    'pooling_method': "diffpool",
    'use_edge_weights': True,
    'dual_stream_aggregation': "concat",
    'network_type': 'timepoints_self' 
}

PROCESSED_DATA_PATH = paths.DATA_ABUNDANCE
FUNCTIONS_PATH = paths.DATA_FUNCTIONAL_ATTR
TAXONOMY_PATH = paths.DATA_PYG / f"HMP_{PARAMS['network_type']}_w_functions_onehot"

# Set up seed-specific output directory
# -----------------------------------------------------
RESULTS_PATH = paths.RES_INTERP_SEEDS / f"seed_{REPRODUCIBLE_SEED}"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


# Load pytorch geometric dataset
# ----------------------------------------------------
DATASET_PATH = TAXONOMY_PATH / f"pyg_dataset_species_threshold_assoc_{PARAMS['t_str']}.pt"
dataset = torch.load(DATASET_PATH, weights_only=False)
valid_graphs = dataset.name

# Load feature tables
# -------------------------------------
func_emb_BP_idf = pd.read_csv(FUNCTIONS_PATH / "HMP" / 
                              "taxa_one_hot_encoded_one_hot_bp_weighted.csv", index_col=0)

# Load taxa hierarchy
taxonomy_onehot = pd.read_csv(TAXONOMY_PATH / f"hierarchy_species_threshold_assoc_{PARAMS['t_str']}.csv", index_col=0)

# load CLR-transformed mean abundances
mean_clr = pd.read_csv(TAXONOMY_PATH / f"clr_values_species_threshold_assoc_{PARAMS['t_str']}.csv", index_col=0).filter(items=valid_graphs, axis=1)

# load CLR variances
var_clr = pd.read_csv(TAXONOMY_PATH / f"clr_variance_species_threshold_assoc_{PARAMS['t_str']}.csv", index_col=0).filter(items=valid_graphs, axis=1)


# Setup class labels
# ------------------------------------------------------
composite_baseline = pd.read_csv(PROCESSED_DATA_PATH / "HMP_counts_filtered_baseline_composite.csv", index_col="subject_id").filter(items=valid_graphs, axis=0)

# Binary: Healthy=1, IBD=0
y_binary_encoded = composite_baseline["condition"].map({"CD": 0, "UC": 0, "healthy": 1}).astype(int)

# Function to concatenate node features based on specified list
# ------------------------------------------------------
def concat_node_features(data, feature_list, include_flags=False): # do not use flags
    x_parts = []

    for feat_name in feature_list:
        # append the feature itself
        x_parts.append(getattr(data, feat_name))

        if include_flags:
            # construct the reliability flag attribute name
            has_name = "has_" + feat_name.split("x_")[-1]
            if hasattr(data, has_name):
                x_parts.append(getattr(data, has_name))

    return torch.cat(x_parts, dim=-1)

# Check input dimension
example_data = dataset[0]
input_dim = concat_node_features(example_data, PARAMS['node_features_list'], include_flags = False).shape[1]
print(f"Input dimension based on selected node features: {input_dim}")


# Model training
# --------------------------------------
train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, 
                                      stratify=[d.y_binary.item() for d in dataset], random_state=REPRODUCIBLE_SEED)

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=8, shuffle=False)

# combine train and val indices and create captum loader
combined_idx = np.concatenate([train_idx, val_idx])
# captum loader with all samples, batch size 1
captum_loader = DataLoader(Subset(dataset, combined_idx), batch_size=1, shuffle=False)

# using alternative model definition and training functions that
# allows for edge weight input in the forward pass (necessary for Integrated Gradients)
model = DualStreamGNN2(
    model_type=PARAMS['model_type'], num_layers=PARAMS['num_layers'],
    input_dim=input_dim, hidden_dim=128, embedding_dim=128, dropout=0.0, # no dropout for clearer interpretability
    num_classes=1, pooling_method=PARAMS['pooling_method'],
    sagpool_ratio=0.5, difpool_clusters_ratio=0.7, aggr_strategy=PARAMS['dual_stream_aggregation']
).to(DEVICE)

# Calculate min/max from current training split
model.fit_global_scaling(train_loader)

optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0).to(DEVICE))

print("Training model...")
trained_model, _ = train_model2(
    model=model, train_loader=train_loader, val_loader=val_loader,
    criterion=criterion, optimizer=optimizer, num_epochs=500, patience=15,
    node_features_list=PARAMS['node_features_list'], threshold=PARAMS['pred_threshold'],
    use_edge_weights=True, y_key="y_binary", plot_losses=False, include_flags=False
)


# Define forward wrapper functions for nodes and edges to be used with Captum's Integrated Gradients.
# ------------------------------------------------------
# Node attribution wrapper
def node_forward_wrapper(x_3d, data_obj, edge_mask):
    # Captum varies x_3d (node features)
    x = x_3d.squeeze(0)
    # Pass the fixed edge_weights (mask) to the model
    logits, _, _ = model(
        data=data_obj, 
        node_features=x, 
        use_edge_weights=True, 
        edge_weights=edge_mask
    )
    # Ensure shape is [batch_size, 1] for Captum
    return logits.view(-1, 1)

# Edge attribution wrapper
def edge_forward_wrapper(mask_3d, data_obj, node_feat):
    # Captum varies mask_3d (raw edge weights)
    mask = mask_3d.squeeze(0)
    # Pass the fixed node_features to the model
    logits, _, _ = model(
        data=data_obj, 
        node_features=node_feat, 
        use_edge_weights=True, 
        edge_weights=mask
    )
    # Ensure shape is [batch_size, 1] for Captum
    return logits.view(-1, 1)

# Initialize Captum objects
ig_nodes = IntegratedGradients(node_forward_wrapper)
ig_edges = IntegratedGradients(edge_forward_wrapper)

# Generate node attributions for each graph in the captum_loader
# ------------------------------------------------------
model.eval()

node_attributions_dict = {}

for data in captum_loader:

    data = data.to(DEVICE)

    x = concat_node_features(data, PARAMS['node_features_list'], include_flags=False)

    # Define baseline as a tensor with zeros
    baseline = torch.zeros_like(x)
    
    # CLR baseline value should not be 0, set to 5th percentile of mean_clr
    global_quantile = mean_clr.values.flatten()
    clr_baseline_value_scalar = np.quantile(global_quantile, 0.05)
    clr_baseline_value = torch.tensor(clr_baseline_value_scalar, device=DEVICE, dtype=x.dtype)

    # CLR variance baseline value should not be 0, set to 5th percentile of var_clr
    # exclude nans from list
    global_var_quantile = var_clr.values.flatten()
    global_var_quantile = global_var_quantile[~np.isnan(global_var_quantile)]
    var_clr_baseline_value_scalar = np.quantile(global_var_quantile, 0.05)
    var_clr_baseline_value = torch.tensor(var_clr_baseline_value_scalar, device=DEVICE, dtype=x.dtype)
    
    # fill the first column (CLR feature) with this value
    baseline[:, 0] = clr_baseline_value
    # fill the second column (CLR variance feature) with this value
    baseline[:, 1] = var_clr_baseline_value

    # Weights
    edge_weights = data.weight

    # Compute Attributions
    # target=None because the output is a scalar (logit).
    attributions, delta = ig_nodes.attribute(
        inputs=x.unsqueeze(0),  # add batch dimension
        baselines=baseline.unsqueeze(0),  # add batch dimension
        additional_forward_args=(data, edge_weights),
        internal_batch_size=1,
        target=None,
        n_steps=200,
        return_convergence_delta=True
    )

    attr_np = attributions.squeeze().cpu().numpy()

    attributions_df = pd.DataFrame(attr_np)
    
    # Setting node names for rows
    attributions_df.index = [node for node in data.node_name[0]]

    # Setting feature names for columns
    # BP_names = func_emb_BP_idf.columns.tolist()
    BP_names = func_emb_BP_idf.columns.tolist()
    taxonomy_names = taxonomy_onehot.columns.tolist()
    clr_name = ["clr"]
    var_name = ["clr_variance"]

    # Combine all feature names in the order they were concatenated
    feature_names = clr_name + var_name + taxonomy_names + BP_names
    attributions_df.columns = feature_names

    # Adding graph name as a column
    attributions_df["graph_name"] = data.name[0]
    # Adding model output (logits) as a column
    attributions_df["logits"] = float(node_forward_wrapper(x.unsqueeze(0), data, edge_weights).detach().cpu().squeeze())
    # Adding true label as a column
    attributions_df["true_label"] = int(data.y_binary[0])
    # Adding delta as a column
    attributions_df["convergence_delta"] = float(delta.detach().cpu().squeeze())

    node_attributions_dict[data.name[0]] = attributions_df


# Generate edge attributions for each graph in the captum_loader
# ------------------------------------------------------
model.eval()

edge_attributions_dict = {}

for data in captum_loader:
    data = data.to(DEVICE)
    x = concat_node_features(data, PARAMS['node_features_list'], include_flags=False)
    
    # create edge mask and baseline
    # data.edge_index.shape[1] gives the number of edges
    num_edges = data.edge_index.shape[1]
    edge_weights = data.weight.unsqueeze(0)  # shape (1, num_edges)
    edge_baseline = torch.zeros((1, num_edges), device=DEVICE)

    # Compute Attributions
    edge_attributions, delta = ig_edges.attribute(
        inputs=edge_weights,  # add batch dimension
        baselines=edge_baseline,  # add batch dimension
        additional_forward_args=(data, x),
        internal_batch_size=1,
        n_steps=200,
        return_convergence_delta=True
    )

    # Process results
    edge_attr_np = edge_attributions.squeeze().cpu().detach().numpy()
    
    # Map back to node names for clarity
    edge_index = data.edge_index.cpu().numpy()
    nodes = data.node_name[0]
    
    # Create a list of "Source -> Target" strings
    edge_labels = [f"{nodes[edge_index[0, i]]}||{nodes[edge_index[1, i]]}" for i in range(num_edges)]

    edge_attr_df = pd.DataFrame({
        "edge": edge_labels,
        "attribution": edge_attr_np
    })

    # Adding graph name as a column
    edge_attr_df["graph_name"] = data.name[0]
    # Adding model output (logits) as a column
    edge_attr_df["logits"] = float(node_forward_wrapper(x.unsqueeze(0), data, edge_weights.squeeze(0)).detach().cpu().squeeze())
    # Adding true label as a column
    edge_attr_df["true_label"] = int(data.y_binary[0])
    # Adding delta as a column
    edge_attr_df["convergence_delta"] = float(delta.detach().cpu().squeeze())
    
    # Store in dictionary
    edge_attributions_dict[data.name[0] + "_edges"] = edge_attr_df

# Save node attribution dataframes
# ---------------------------------------------

# Save separate CSVs per graph
for graph_name, attr_df in node_attributions_dict.items():
    attr_df.to_csv(RESULTS_PATH / f"integrated_gradients_attributions_{graph_name}.csv")

for graph_name, edge_attr_df in edge_attributions_dict.items():
    edge_attr_df.to_csv(RESULTS_PATH / f"integrated_gradients_edge_attributions_{graph_name}.csv")

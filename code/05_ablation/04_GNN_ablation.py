import copy
from collections import defaultdict
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import re
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import train_test_split

# --- CUSTOM IMPORTS ---
from graph_model_functions import (
    train_model,
    validate_model,
    extract_embeddings_df2,
    get_graph_names_and_labels,
    set_seed,
)

from model_definitions import (
    FlexibleGNN,
    DualStreamGNN
)

# --- EXPERIMENT CONFIGURATION & PATHS ---
PROJECT_ROOT = Path.cwd()
PROCESSED_DATA_PATH = PROJECT_ROOT / "data/processed/"

# Path to module membership table relative to project structure
TABLE_PATH = PROJECT_ROOT / "results/tables/analyses/attributions/node_module_membership_table.csv"

EXPERIMENTAL_CONFIG = {
    "num_repeats": 30,
    "validation_fraction": 0.2,
    "dropout_rate": 0.3,
    "num_epochs": 500,
    "patience": 50,
    "learning_rate": 5e-6,
    "batch_size": 8,
    "reproducible_seed": 500,
}

MODEL_PARAMS = {
    't_str': "0_3",
    'training_dataset': 'HMP',
    'validation_dataset': 'Hall',
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


# --- HELPER FUNCTIONS ---
def clean_node_name(name_str):
    if not isinstance(name_str, str):
        name_str = str(name_str)
    cleaned = re.sub(r".*s__", "", name_str)
    cleaned = cleaned.replace("_", " ").strip().lower()
    return cleaned


def load_module_species(table_path):
    """Loads species dynamically from the provided module membership CSV table."""
    df = pd.read_csv(table_path)
    
    m1_species = df[df['module'] == 'M1']['species_name'].dropna().tolist()
    m2_species = df[df['module'] == 'M2']['species_name'].dropna().tolist()
    
    return m1_species, m2_species


def filter_pyg_dataset(dataset, target_nodes=None, exclude_nodes=None):
    """
    Filters PyG graphs based on allowed (target_nodes) and/or excluded (exclude_nodes) species.
    """
    target_clean = {clean_node_name(n) for n in target_nodes} if target_nodes is not None else None
    exclude_clean = {clean_node_name(n) for n in exclude_nodes} if exclude_nodes is not None else set()

    filtered_dataset = []

    for data in dataset:
        if not hasattr(data, "node_name"):
            raise AttributeError("Data object has no 'node_name' attribute.")

        node_names = data.node_name
        if torch.is_tensor(node_names):
            node_names = node_names.tolist()

        node_names = [str(x) for x in node_names]

        keep_indices = []
        for i, name in enumerate(node_names):
            cleaned = clean_node_name(name)
            if exclude_nodes and cleaned in exclude_clean:
                continue
            if target_clean is not None and cleaned not in target_clean:
                continue
            keep_indices.append(i)

        if not keep_indices:
            continue

        keep_indices_tensor = torch.tensor(
            keep_indices,
            dtype=torch.long,
            device=data.edge_index.device
        )

        old_to_new = torch.full(
            (data.num_nodes,),
            -1,
            dtype=torch.long,
            device=data.edge_index.device
        )
        old_to_new[keep_indices_tensor] = torch.arange(
            len(keep_indices),
            device=data.edge_index.device
        )

        edge_index = data.edge_index
        src, dst = edge_index[0], edge_index[1]
        edge_mask = (old_to_new[src] >= 0) & (old_to_new[dst] >= 0)

        new_edge_index = torch.stack([
            old_to_new[src[edge_mask]],
            old_to_new[dst[edge_mask]]
        ], dim=0)

        filtered_data = copy.copy(data)

        # Filter node-level attributes
        for key in data.keys():
            value = data[key]
            if key == "edge_index":
                continue

            if (
                torch.is_tensor(value)
                and value.ndim > 0
                and value.size(0) == data.num_nodes
            ):
                filtered_data[key] = value[keep_indices_tensor]

        # Filter edge-level attributes
        num_edges = data.edge_index.size(1)
        for key in data.keys():
            value = data[key]
            if (
                key != "edge_index"
                and torch.is_tensor(value)
                and value.ndim > 0
                and value.size(0) == num_edges
            ):
                filtered_data[key] = value[edge_mask]

        filtered_data.edge_index = new_edge_index
        filtered_data.node_name = [node_names[i] for i in keep_indices]
        filtered_data.node_names = filtered_data.node_name
        filtered_data.num_nodes = len(keep_indices)

        filtered_dataset.append(filtered_data)

    print(f"Filtered dataset: {len(dataset)} -> {len(filtered_dataset)} graphs")
    return filtered_dataset


# --- MAIN EXPERIMENT PIPELINE ---
def main():
    set_seed(EXPERIMENTAL_CONFIG["reproducible_seed"])
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on device: {DEVICE}")

    # Load dynamic module species from table
    m1_species, m2_species = load_module_species(TABLE_PATH)
    m1_plus_m2_species = list(set(m1_species + m2_species))

    # Define target species subsets (positive inclusion filters)
    target_sets = {
        "m1": m1_species,
        "m2": m2_species,
        "m1_plus_m2": m1_plus_m2_species,
    }

    # Define exclusion lists built off whole dataset
    exclusions = {
        "whole_minus_m1": m1_species,
        "whole_minus_m2": m2_species,
        "whole_minus_m1_plus_m2": m1_plus_m2_species,
    }

    # 1. LOAD BASELINE WHOLE DATASETS
    dataset_name = f"pyg_dataset_species_threshold_{MODEL_PARAMS['weight_type']}{MODEL_PARAMS['t_str']}.pt"

    dataset_path_train = PROCESSED_DATA_PATH / "pyg_datasets" / \
        f"{MODEL_PARAMS['training_dataset']}_{MODEL_PARAMS['network_type']}_w_functions_onehot" / dataset_name

    dataset_path_hall = PROCESSED_DATA_PATH / "pyg_datasets" / \
        f"{MODEL_PARAMS['validation_dataset']}_{MODEL_PARAMS['network_type']}_w_functions_onehot" / dataset_name

    print(f"Loading training dataset from: {dataset_path_train}")
    raw_train_dataset = torch.load(dataset_path_train, weights_only=False)
    
    print(f"Loading external dataset from: {dataset_path_hall}")
    raw_hall_dataset = torch.load(dataset_path_hall, weights_only=False)

    # 2. PRE-COMPUTE AND CACHE ALL EXPERIMENTAL GRAPH VARIANTS
    print("\n--- Constructing and Caching Graph Variants ---")
    datasets_train_suite = {"whole": raw_train_dataset}
    datasets_hall_suite = {"whole": raw_hall_dataset}

    # Positive inclusion filters
    for name, species_list in target_sets.items():
        print(f"Processing Filtered Set: '{name}'")
        datasets_train_suite[name] = filter_pyg_dataset(raw_train_dataset, target_nodes=species_list)
        datasets_hall_suite[name] = filter_pyg_dataset(raw_hall_dataset, target_nodes=species_list)

    # Exclusion filters
    for name, excl_list in exclusions.items():
        print(f"Processing Exclusion Set: '{name}'")
        datasets_train_suite[name] = filter_pyg_dataset(raw_train_dataset, exclude_nodes=excl_list)
        datasets_hall_suite[name] = filter_pyg_dataset(raw_hall_dataset, exclude_nodes=excl_list)

    training_configs = list(datasets_train_suite.keys())
    evaluation_sets = list(datasets_train_suite.keys())

    # Pre-extract label vectors for stratified splitting based on whole dataset indices
    labels = np.array([data.y_binary.item() for data in raw_train_dataset])
    all_dataset_indices = np.arange(len(raw_train_dataset))

    metrics_storage = defaultdict(list)
    raw_runs_list = []          # List to track individual run summary metrics
    all_individual_preds_df = [] # List to track sample-level predictions

    # 3. MONTE CARLO EXPERIMENT LOOP
    print(f"\n--- Starting Cross-Validation Pipeline ({EXPERIMENTAL_CONFIG['num_repeats']} Repeats) ---")

    for repeat_idx in range(EXPERIMENTAL_CONFIG["num_repeats"]):
        print(f"\n==================== REPEAT RUN {repeat_idx + 1}/{EXPERIMENTAL_CONFIG['num_repeats']} ====================")

        train_indices, val_indices = train_test_split(
            all_dataset_indices, 
            test_size=EXPERIMENTAL_CONFIG["validation_fraction"], 
            stratify=labels, 
            random_state=EXPERIMENTAL_CONFIG["reproducible_seed"] + repeat_idx 
        )

        # Record split metadata for tracing
        train_names_df = get_graph_names_and_labels(raw_train_dataset, [all_dataset_indices[i] for i in train_indices])
        val_names_df = get_graph_names_and_labels(raw_train_dataset, [all_dataset_indices[i] for i in val_indices])
        split_df = pd.concat([train_names_df.assign(split='train'), val_names_df.assign(split='val')], ignore_index=True)
        split_df['fold'] = repeat_idx + 1

        # Train a separate model for each training configuration
        for train_config in training_configs:
            print(f"\n>>> Training Model on Dataset Variant: [{train_config.upper()}]")
            train_ds = datasets_train_suite[train_config]

            train_subset = Subset(train_ds, train_indices)
            val_subset = Subset(train_ds, val_indices)

            train_loader = PyGDataLoader(train_subset, batch_size=EXPERIMENTAL_CONFIG["batch_size"], shuffle=True)
            val_loader = PyGDataLoader(val_subset, batch_size=EXPERIMENTAL_CONFIG["batch_size"], shuffle=False)

            # Compute Input Feature Dimension
            input_dim = sum(
                getattr(train_ds[0], feat_name).shape[-1] 
                for feat_name in MODEL_PARAMS['node_features_list']
            )

            # Build Model Architecture
            if not MODEL_PARAMS['use_edge_weights']:
                model = FlexibleGNN(
                    model_type=MODEL_PARAMS['model_type'], num_layers=MODEL_PARAMS['num_layers'],
                    input_dim=input_dim, hidden_dim=128, embedding_dim=128, dropout=EXPERIMENTAL_CONFIG["dropout_rate"],
                    num_classes=1, pooling_method=MODEL_PARAMS['pooling_method'],
                    sagpool_ratio=0.5, difpool_clusters_ratio=0.7
                ).to(DEVICE)
            else:
                model = DualStreamGNN(
                    model_type=MODEL_PARAMS['model_type'], num_layers=MODEL_PARAMS['num_layers'],
                    input_dim=input_dim, hidden_dim=128, embedding_dim=128, dropout=EXPERIMENTAL_CONFIG["dropout_rate"],
                    num_classes=1, pooling_method=MODEL_PARAMS['pooling_method'],
                    sagpool_ratio=0.5, difpool_clusters_ratio=0.7, aggr_strategy=MODEL_PARAMS['dual_stream_aggregation']
                ).to(DEVICE)
                model.fit_global_scaling(train_loader)

            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0).to(DEVICE)).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=EXPERIMENTAL_CONFIG["learning_rate"])

            # Train Model on Selected Strategy Configuration
            trained_model, _ = train_model(
                model=model, train_loader=train_loader, val_loader=val_loader,
                criterion=criterion, optimizer=optimizer, num_epochs=EXPERIMENTAL_CONFIG["num_epochs"],
                patience=EXPERIMENTAL_CONFIG["patience"], node_features_list=MODEL_PARAMS['node_features_list'],
                threshold=MODEL_PARAMS['pred_threshold'],
                use_edge_weights=MODEL_PARAMS['use_edge_weights'], y_key="y_binary", plot_losses=False,
                repeat_idx=repeat_idx + 1
            )

            # 4. CROSS-EVALUATION ACROSS ALL EVALUATION SETS
            print(f"--- Evaluating [{train_config.upper()}] model across ALL evaluation sets ---")
            for eval_set_name in evaluation_sets:
                # A. Internal Validation Evaluation
                eval_val_ds = Subset(datasets_train_suite[eval_set_name], val_indices)
                eval_val_loader = PyGDataLoader(eval_val_ds, batch_size=EXPERIMENTAL_CONFIG["batch_size"], shuffle=False)

                val_res, _, _, _ = validate_model(
                    model=trained_model, loader=eval_val_loader, criterion=criterion,
                    node_features_list=MODEL_PARAMS['node_features_list'],
                    use_edge_weights=MODEL_PARAMS['use_edge_weights'], y_key="y_binary",
                    threshold=MODEL_PARAMS['pred_threshold'], device=DEVICE
                )

                # Extract individual predictions for Internal Validation
                eval_val_extraction_loader = PyGDataLoader(eval_val_ds, batch_size=1, shuffle=False)
                val_graph_df, _ = extract_embeddings_df2(
                    model=trained_model, loader=eval_val_extraction_loader,
                    node_features_list=MODEL_PARAMS['node_features_list'],
                    use_edge_weights=MODEL_PARAMS['use_edge_weights'],
                    threshold=MODEL_PARAMS['pred_threshold'],
                    y_key="y_binary", device=DEVICE
                )
                
                if val_graph_df is not None and not val_graph_df.empty:
                    val_graph_df = val_graph_df.rename(columns={'label': 'true_label', 'preds': 'predicted_label'})
                    val_graph_df['repeat_idx'] = repeat_idx + 1
                    val_graph_df['train_config'] = train_config
                    val_graph_df['eval_set'] = eval_set_name
                    val_graph_df['split'] = 'internal_validation'
                    val_graph_df['dataset_origin'] = MODEL_PARAMS['training_dataset']
                    all_individual_preds_df.append(val_graph_df)

                # B. External (Hall) Evaluation
                eval_hall_ds = datasets_hall_suite[eval_set_name]
                eval_hall_loader = PyGDataLoader(eval_hall_ds, batch_size=1, shuffle=False)

                hall_res, _, _, _ = validate_model(
                    model=trained_model, loader=eval_hall_loader, criterion=criterion,
                    node_features_list=MODEL_PARAMS['node_features_list'],
                    use_edge_weights=MODEL_PARAMS['use_edge_weights'], y_key="y_binary",
                    threshold=MODEL_PARAMS['pred_threshold'], device=DEVICE
                )

                # Extract individual predictions for External Hall Dataset
                hall_graph_df, _ = extract_embeddings_df2(
                    model=trained_model, loader=eval_hall_loader,
                    node_features_list=MODEL_PARAMS['node_features_list'],
                    use_edge_weights=MODEL_PARAMS['use_edge_weights'],
                    threshold=MODEL_PARAMS['pred_threshold'],
                    y_key="y_binary", device=DEVICE
                )

                if hall_graph_df is not None and not hall_graph_df.empty:
                    hall_graph_df = hall_graph_df.rename(columns={'label': 'true_label', 'preds': 'predicted_label'})
                    hall_graph_df['repeat_idx'] = repeat_idx + 1
                    hall_graph_df['train_config'] = train_config
                    hall_graph_df['eval_set'] = eval_set_name
                    hall_graph_df['split'] = 'external_validation'
                    hall_graph_df['dataset_origin'] = MODEL_PARAMS['validation_dataset']
                    all_individual_preds_df.append(hall_graph_df)

                metrics_dict = {
                    "internal_loss": val_res[0], "internal_acc": val_res[1],
                    "internal_auc": val_res[2],  "internal_auprc": val_res[3], "internal_f1": val_res[6],
                    "external_loss": hall_res[0], "external_acc": hall_res[1],
                    "external_auc": hall_res[2],  "external_auprc": hall_res[3], "external_f1": hall_res[6]
                }

                # Append to aggregated storage
                key = (train_config, eval_set_name)
                metrics_storage[key].append(metrics_dict)

                # Store per-repeat raw row
                raw_row = {
                    "repeat_idx": repeat_idx + 1,
                    "train_config": train_config,
                    "eval_set": eval_set_name,
                    **metrics_dict
                }
                raw_runs_list.append(raw_row)

    # 5. AGGREGATE RESULTS & COMPILE COMPREHENSIVE TABLES
    summary_rows = []
    metric_keys = ["auc", "auprc", "f1", "loss", "acc"]

    for (tr_cfg, ev_set), runs in metrics_storage.items():
        row = {
            "train_config": tr_cfg,
            "eval_set": ev_set
        }

        # Compute Mean and Std for both Internal Val and External Hall across repeats
        for dataset_type in ["internal", "external"]:
            for metric in metric_keys:
                metric_vals = [run[f"{dataset_type}_{metric}"] for run in runs]
                row[f"{dataset_type}_{metric}_mean"] = np.mean(metric_vals)
                row[f"{dataset_type}_{metric}_std"] = np.std(metric_vals)

        summary_rows.append(row)

    results_df = pd.DataFrame(summary_rows)
    raw_results_df = pd.DataFrame(raw_runs_list)
    final_predictions_df = pd.concat(all_individual_preds_df, ignore_index=True)

    # Save outputs locally with updated filenames
    output_path = PROJECT_ROOT / "cross_evaluation_performance_summary_modules.csv"
    raw_output_path = PROJECT_ROOT / "cross_evaluation_raw_metrics_per_run_modules.csv"
    preds_output_path = PROJECT_ROOT / "cross_evaluation_predictions_modules.csv"

    # results_df.to_csv(output_path, index=False)
    # raw_results_df.to_csv(raw_output_path, index=False)
    # final_predictions_df.to_csv(preds_output_path, index=False)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE: FULL PERFORMANCE SUMMARY TABLE (MODULE COMBINATIONS)")
    print("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # Display formatted core metric columns
    display_cols = [
        "train_config", "eval_set",
        "internal_auc_mean", "internal_auprc_mean", "internal_f1_mean",
        "external_auc_mean", "external_auprc_mean", "external_f1_mean"
    ]
    print(results_df[display_cols].to_string(index=False))
    print(f"\nSaved comprehensive cross-evaluation summary table to:\n{output_path}")
    print(f"Saved raw per-repeat metrics table to:\n{raw_output_path}")
    print(f"Saved individual sample-level predictions table to:\n{preds_output_path}\n")


if __name__ == "__main__":
    main()
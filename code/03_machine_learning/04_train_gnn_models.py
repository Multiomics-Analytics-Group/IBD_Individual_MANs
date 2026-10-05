# RUN THIS FROM THE TERMINAL AT THE CURRENT DIRECTORY (code/03_machine_learning) TO ENSURE PROPER PATH RESOLUTION

from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import ParameterGrid
from pathlib import Path
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from collections import defaultdict
import numpy as np
import pickle
import sys

# Locate the project root (works from the terminal and interactively) and load shared paths
_start = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
_root = next(p for p in (_start, *_start.parents) if (p / "code" / "common" / "paths.py").exists())
sys.path.insert(0, str(_root / "code" / "common"))
import paths
import helper_functions  # noqa: F401  (defines GraphDictDataset, needed by torch.load to read the saved datasets)

# Import custom functions
from graph_model_functions import (
    train_model,
    validate_model,
    extract_embeddings_df,
    collect_dataset_stats_by_label,
    set_seed,
    get_graph_names_and_labels
)

from model_definitions import (
        FlexibleGNN,
        DualStreamGNN
)

RESULTS_PATH = paths.RES_GNN_PICKLES

REPRODUCIBLE_SEED = 200
set_seed(REPRODUCIBLE_SEED)


# --- PARAMETER GRID DEFINITION ---

param_grid = {
    'network_type': ['timepoints_self'],
    't_str': ["0_3", "0_4", "0_5"],
    'pooling_method': ["diffpool"],
    'node_features_list': [

        # solo feature sets
        ['x_dummy'],
        ['x_mean_clr', 'x_clr_variance'],
        ['x_taxonomy_onehot'],
        ['x_func_emb_BP_idf'],

        # combinations
        ['x_mean_clr', 'x_clr_variance', 'x_taxonomy_onehot'],
        ['x_mean_clr', 'x_clr_variance', 'x_func_emb_BP_idf'],
        ['x_taxonomy_onehot', 'x_func_emb_BP_idf'],
        ['x_mean_clr', 'x_clr_variance', 'x_taxonomy_onehot', 'x_func_emb_BP_idf']

    ],
    'training_dataset': ['HMP'],
    'validation_dataset': ['Hall'],
    'model_type': ['GraphConv'],
    'weight_type': ['assoc_'],
    'num_layers': [2],
    'pred_threshold': [0.5],
    'use_edge_weights': [False, True],
    'dual_stream_aggregation': ["concat", "subtract"]
}


# Generate all combinations
grid_combinations = list(ParameterGrid(param_grid))
TOTAL_GRIDS = len(grid_combinations)
print(f"Total experiment combinations: {TOTAL_GRIDS}")

# --- HYPERPARAMETERS AND SETUP ---
NUM_REPEATS = 50
VALIDATION_FRACTION = 0.2
DROPOUT_RATE = 0.3
NUM_EPOCHS = 500
PATIENCE = 50
LEARNING_RATE = 0.000005
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if isinstance(RESULTS_PATH, str):
    RESULTS_PATH = Path(RESULTS_PATH)

# --- EXPERIMENT STORAGE ---
all_experiment_results = []

# --- OUTER GRID SEARCH LOOP ---
for grid_idx, params in enumerate(grid_combinations):
    print(f"\n=======================================================")
    print(f"Starting Experiment {grid_idx + 1}/{TOTAL_GRIDS} with parameters:")
    print(params)
    print(f"=======================================================")

    # 1. DATA LOADING
    # Training data
    dataset_name = ("pyg_dataset_species_threshold_"
                    f"{params['weight_type']}{params['t_str']}.pt")

    dataset_path = paths.DATA_PYG / \
        (f'{params['training_dataset']}_{params['network_type']}_w_functions_onehot') / dataset_name

    # External data
    dataset_path_Hall = paths.DATA_PYG / \
        (f'{params['validation_dataset']}_{params["network_type"]}_w_functions_onehot') / dataset_name


    dataset = torch.load(dataset_path, weights_only=False)
    dataset_stats_by_label = collect_dataset_stats_by_label(dataset)

    dataset_Hall = torch.load(dataset_path_Hall, weights_only=False)
    dataset_stats_by_label_Hall = collect_dataset_stats_by_label(dataset_Hall)
    Hall_loader = PyGDataLoader(dataset_Hall, batch_size=1, shuffle=False)

    # 2. MONTE CARLO SETUP
    labels = np.array([data.y_binary.item() for data in dataset])
    all_dataset_indices = np.arange(len(dataset))

    fold_metrics = defaultdict(list)
    curve_data_by_repeat = {} 
    all_fold_run_summaries = [] 
    
    all_graph_embeddings_dfs = [] 
    all_node_embeddings_dfs = []
    all_split_info = []

    # MONTE CARLO LOOP (Repeated Splits)
    for repeat_idx in range(NUM_REPEATS):
        print(f"\n--- Starting Monte Carlo Run {repeat_idx + 1}/{NUM_REPEATS} ---")

        # --- DATA SPLIT ---
        train_indices, val_indices = train_test_split(
            all_dataset_indices, test_size=VALIDATION_FRACTION, 
            stratify=labels, random_state=REPRODUCIBLE_SEED + repeat_idx 
        )
        train_names_df = get_graph_names_and_labels(dataset, [all_dataset_indices[i] for i in train_indices])
        val_names_df = get_graph_names_and_labels(dataset, [all_dataset_indices[i] for i in val_indices])
        
        split_df = pd.concat([train_names_df.assign(split='train'), val_names_df.assign(split='val')], ignore_index=True)
        split_df['fold'] = repeat_idx + 1
        all_split_info.append(split_df)
        
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices) 

        train_loader = PyGDataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = PyGDataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
        val_loader_extraction = PyGDataLoader(val_subset, batch_size=1, shuffle=False)

        # 3. MODEL INITIALIZATION AND TRAINING
        input_dim = 0
        for feat_name in params['node_features_list']:
            input_dim += getattr(dataset[0], feat_name).shape[-1]
          #  has_name = "has_" + feat_name.split("x_")[-1]
          #  if hasattr(dataset[0], has_name):
          #      input_dim += getattr(dataset[0], has_name).shape[-1]

        if not params['use_edge_weights']:
            # If not using edge weights, use a simple model without the dual stream architecture,
            # as edge weights won't be processed in a separate stream.
            model = FlexibleGNN(
                model_type=params['model_type'], num_layers=params['num_layers'],
                input_dim=input_dim, hidden_dim=128, embedding_dim=128, dropout=DROPOUT_RATE,
                num_classes=1, pooling_method=params['pooling_method'],
                sagpool_ratio=0.5, difpool_clusters_ratio=0.7
            ).to(DEVICE)
        else:
            model = DualStreamGNN(
                model_type=params['model_type'], num_layers=params['num_layers'],
                input_dim=input_dim, hidden_dim=128, embedding_dim=128, dropout=DROPOUT_RATE,
                num_classes=1, pooling_method=params['pooling_method'],
                sagpool_ratio=0.5, difpool_clusters_ratio=0.7, aggr_strategy=params['dual_stream_aggregation']
            ).to(DEVICE)
            # Calculate min/max from current training split
            model.fit_global_scaling(train_loader)


        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0).to(DEVICE)).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        trained_model, history = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer, num_epochs=NUM_EPOCHS,
            patience=PATIENCE, node_features_list=params['node_features_list'],
            threshold=params['pred_threshold'],
            use_edge_weights=params['use_edge_weights'], y_key="y_binary", plot_losses=False,
            repeat_idx=repeat_idx + 1
        )
        
        # EVALUATE ON EXTERNAL HALL DATA
        hall_metrics_results, _, _, _ = validate_model(
            model=trained_model,
            loader=Hall_loader,
            criterion=criterion,
            node_features_list=params['node_features_list'],
            use_edge_weights=params['use_edge_weights'],
            y_key="y_binary",
            threshold=params['pred_threshold'],
            device=DEVICE
        )

        h_loss, h_acc, h_auc, h_auprc, h_prec, h_rec, h_f1 = hall_metrics_results[:7]

        # Store Hall metrics in fold_metrics for aggregation
        fold_metrics["hall_loss"].append(h_loss)
        fold_metrics["hall_auc"].append(h_auc)
        fold_metrics["hall_auprc"].append(h_auprc)
        fold_metrics["hall_f1"].append(h_f1)
        fold_metrics["hall_acc"].append(h_acc)

        # 4. COLLECT METRICS AND EMBEDDINGS
        fold_run_summary = {
            'grid_idx': grid_idx + 1, 'fold_idx': repeat_idx + 1, 'params': params,
            'dataset_stats': dataset_stats_by_label, 'split_info': split_df.to_dict('list'),
            'raw_history': history,
        }

        if history["val_loss"]:
            best_epoch_idx = np.argmin(history["val_loss"])

            SCALAR_METRICS = ["val_rec", "val_prec", "val_loss", "val_acc", "val_auc", "val_auprc", "val_f1"]
            for metric in SCALAR_METRICS:
                value_to_append = history[metric][best_epoch_idx]
                if hasattr(value_to_append, 'item'): scalar_value = value_to_append.item()
                elif isinstance(value_to_append, (list, np.ndarray)) and len(value_to_append) == 1: scalar_value = float(value_to_append[0])
                else: scalar_value = float(value_to_append)
                    
                fold_metrics[metric].append(scalar_value)
                fold_run_summary[f"best_val_{metric}"] = scalar_value

            CURVE_METRICS = ["fpr", "tpr", "thresh_roc", "prec", "rec", "thresh_prc"]
            curve_data_by_repeat[repeat_idx] = {metric: history[metric][best_epoch_idx] for metric in CURVE_METRICS}
            fold_run_summary['best_curve_data'] = curve_data_by_repeat[repeat_idx]

            
            graph_df, node_df = extract_embeddings_df(
                model=trained_model, loader=val_loader_extraction,
                node_features_list=params['node_features_list'], use_edge_weights=params['use_edge_weights'],
                threshold=params['pred_threshold'],
                y_key="y_binary", device=DEVICE
            )
            graph_df = graph_df.rename(columns={'label': 'true_label', 'preds': 'predicted_label'})
            graph_df['dataset_origin'] = 'HMP'

            # --- EXTRACT HALL EMBEDDINGS  ---
            hall_extraction_loader = PyGDataLoader(dataset_Hall, batch_size=1, shuffle=False)
            hall_graph_df, _ = extract_embeddings_df(
                model=trained_model, loader=hall_extraction_loader,
                node_features_list=params['node_features_list'], use_edge_weights=params['use_edge_weights'],
                threshold=params['pred_threshold'],
                y_key="y_binary", device=DEVICE
            )
            hall_graph_df = hall_graph_df.rename(columns={'label': 'true_label', 'preds': 'predicted_label'})
            hall_graph_df['dataset_origin'] = 'Hall'

            # --- MERGE AND STORE ---
            hmp_preds = graph_df.assign(fold=repeat_idx + 1)
            merged_hmp = pd.merge(
                split_df[split_df['split'] == 'val'].drop(columns=['true_label']),
                hmp_preds, on=['graph_name', 'fold'], how='inner' 
            )
            
            # For Hall: All samples are external, so split is fixed
            hall_preds = hall_graph_df.assign(fold=repeat_idx + 1, split='external_validation')
            
            # Combine both into the tracking list
            combined_run_df = pd.concat([merged_hmp, hall_preds], ignore_index=True)
            all_graph_embeddings_dfs.append(combined_run_df)

            # Keep node embeddings logic for HMP
            if node_df is not None and not node_df.empty:
                node_df['fold'] = repeat_idx + 1 
                node_df['grid_idx'] = grid_idx + 1 
                all_node_embeddings_dfs.append(node_df)

            all_fold_run_summaries.append(fold_run_summary)
            
    # AGGREGATE METRICS
    val_auprc_values = fold_metrics.get("val_auprc", [])
    if not val_auprc_values:
        print("Skipping aggregation: No valid repeats completed.")
        continue

    results_row = params.copy()
    results_row.update(dataset_stats_by_label)

    hall_stats = {f"hall_{k}": v for k, v in dataset_stats_by_label_Hall.items()}
    results_row.update(hall_stats)

    for metric, values in fold_metrics.items():
        if values:
            # Save the summarized values (mean and std)
            results_row[f"mean_{metric}"] = np.mean(values)
            results_row[f"std_{metric}"] = np.std(values)
            # Save the raw list of per-fold values
            results_row[f"raw_{metric}"] = values
    
    for metrics in ["hall_loss", "hall_auc", "hall_auprc", "hall_f1", "hall_acc"]:
        if metrics in fold_metrics and fold_metrics[metrics]:
            results_row[f"mean_{metrics}"] = np.mean(fold_metrics[metrics])
            results_row[f"std_{metrics}"] = np.std(fold_metrics[metrics])
            results_row[f"raw_{metrics}"] = fold_metrics[metrics]

    results_row['all_curve_data'] = curve_data_by_repeat
    results_row['all_fold_summaries'] = all_fold_run_summaries

    all_experiment_results.append(results_row)

    # Create output path if it doesn't exist
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # SAVE ARTIFACTS (PICKLE 1: EMBEDDINGS/PREDICTIONS)
    final_val_predictions_df = pd.concat(all_graph_embeddings_dfs, ignore_index=True)
    final_node_embeddings_df = pd.concat(all_node_embeddings_dfs, ignore_index=True)

    final_split_info_df = pd.concat(all_split_info, ignore_index=True)
    
    final_split_table = pd.merge(
        final_split_info_df,
        final_val_predictions_df[['graph_name', 'fold', 'predicted_label']],
        on=['graph_name', 'fold'], how='outer'
    )
    
    final_split_table = final_split_table[['fold', 'split', 'graph_name', 'true_label', 'predicted_label']]


    SPECIFIC_RESULTS_PATH = RESULTS_PATH
    SPECIFIC_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    embeddings_output_path = SPECIFIC_RESULTS_PATH / f"artifacts_grid_{grid_idx+1}_COMBINED.pkl" 
    with open(embeddings_output_path, 'wb') as f:
        pickle.dump({
            'graph_level_embeddings': final_val_predictions_df, 
            'node_level_embeddings': final_node_embeddings_df, 
            'full_split_table': final_split_table
        }, f)
    print(f"   Saved combined artifacts (Pickle 1/2) to: {embeddings_output_path.name}")

    # SAVE SUMMARY (PICKLE 2: METRICS/METADATA)
    summary_output_path = SPECIFIC_RESULTS_PATH / f"summary_grid_{grid_idx+1}_AGGREGATED.pkl"
    with open(summary_output_path, 'wb') as f:
        pickle.dump(results_row, f)
    print(f"   Saved aggregated summary (Pickle 2/2) to: {summary_output_path.name}")

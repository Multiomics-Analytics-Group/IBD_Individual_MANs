import pandas as pd
import pickle
import sys
from pathlib import Path
import re
import numpy as np

# Function for Traditional Model Pickles
def prepare_traditional_for_r(traditional_folder_path, save_curves_csv=True):
    results_path = Path(traditional_folder_path)
    if not results_path.is_dir():
        print(f"Error: traditional Folder not found at {traditional_folder_path}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_summaries = []
    all_sample_tracking = []
    all_curve_rows = []
    all_features = []

    traditional_pattern = re.compile(r'(\w+)_(\w+)_(\w+)_MCarlo\.pkl')

    for file_path in results_path.iterdir():
        match = traditional_pattern.match(file_path.name)
        if not match: 
            continue

        dataset_origin, tax_rank, label_type = match.groups()

        with open(file_path, 'rb') as f:
            model_runs = pickle.load(f)

        for model_name, repeats in model_runs.items():
            for repeat_data in repeats:
                run_id = repeat_data['run']

                # METRICS (Internal & External) 
                # HMP Internal
                summary_hmp = {
                    'dataset_origin': dataset_origin,
                    'tax_rank': tax_rank,
                    'label_type': label_type,
                    'model': model_name,
                    'run': run_id,
                    'val_type': 'internal_hmp',
                    **repeat_data['metrics']
                }
                all_summaries.append(summary_hmp)

                # Hall External
                if 'metrics_external' in repeat_data and repeat_data['metrics_external'] is not None:
                    summary_hall = {
                        'dataset_origin': dataset_origin,
                        'tax_rank': tax_rank,
                        'label_type': label_type,
                        'model': model_name,
                        'run': run_id,
                        'val_type': 'external_hall',
                        **repeat_data['metrics_external']
                    }
                    all_summaries.append(summary_hall)

                # SAMPLE TRACKING (Predictions) 
                if 'sample_tracking' in repeat_data:
                    df_hmp = repeat_data['sample_tracking'].copy()
                    df_hmp['val_type'] = 'internal_hmp'
                    
                    if 'sample_tracking_external' in repeat_data and repeat_data['sample_tracking_external'] is not None:
                        df_hall = repeat_data['sample_tracking_external'].copy()
                        df_hall['val_type'] = 'external_hall'
                        df_tracking = pd.concat([df_hmp, df_hall], ignore_index=True)
                    else:
                        df_tracking = df_hmp
                    
                    df_tracking['model'] = model_name
                    df_tracking['run'] = run_id
                    all_sample_tracking.append(df_tracking)

                # CURVES (Tidy Format for R) 
                if save_curves_csv:
                    for key in ['fpr_micro', 'tpr_micro', 'prec_micro', 'rec_micro', 
                                'fpr_macro', 'tpr_macro', 'prec_macro', 'rec_macro']:
                        val = repeat_data.get(key)
                        if val is not None:
                            all_curve_rows.append({
                                'model': model_name,
                                'run': run_id,
                                'val_type': 'internal_hmp',
                                'metric_name': key,
                                'values': val.tolist() if isinstance(val, np.ndarray) else val
                            })

                if 'selected_features' in repeat_data:
                    features = repeat_data['selected_features']
                    # Check if importance scores exist (RF)
                    importances = repeat_data.get('feature_importances')
                    
                    for i, feat_name in enumerate(features):
                        feat_row = {
                            'model': model_name,
                            'run': run_id,
                            'tax_rank': tax_rank,
                            'feature': feat_name,
                            'importance_score': importances[i] if importances is not None else np.nan,
                            'selected': 1
                        }

                        all_features.append(feat_row)

    summary_df = pd.DataFrame(all_summaries)
    tracking_df = pd.concat(all_sample_tracking, ignore_index=True) if all_sample_tracking else pd.DataFrame()
    
    # Explode curves into a long format
    if all_curve_rows:
        curves_df = pd.DataFrame(all_curve_rows).explode('values')
        curves_df['point_idx'] = curves_df.groupby(['model', 'run', 'metric_name']).cumcount()
    else:
        curves_df = pd.DataFrame()

    features_df = pd.DataFrame(all_features)

    return summary_df, tracking_df, curves_df, features_df

#  Function for Graph Model Pickles 
def prepare_graph_for_r(graph_folder_path, save_curve_csv=True):
    results_path = Path(graph_folder_path)
    if not results_path.is_dir():
        print(f"Error: Graph Folder not found at {graph_folder_path}")
        # FIXED: Return 3 items to match the successful return
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_summaries = []
    all_graph_embeddings = []
    all_curve_rows = []

    summary_pattern = re.compile(r'summary_grid_(\d+)_AGGREGATED\.pkl')

    for file_path in results_path.iterdir():
        match = summary_pattern.match(file_path.name)
        if not match: continue
        
        grid_idx = int(match.group(1))
        with open(file_path, 'rb') as f:
            summary_data = pickle.load(f)

        # METRICS
        hmp_row = {k: v for k, v in summary_data.items() if "hall" not in k and k not in ['all_fold_summaries', 'all_curve_data']}
        hmp_row['val_type'] = 'internal_hmp'
        hmp_row['grid_idx'] = grid_idx
        all_summaries.append(hmp_row)

        hall_row = {k.replace('hall_', ''): v for k, v in summary_data.items() if "hall" in k}
        if hall_row:
            hall_row['val_type'] = 'external_hall'
            hall_row['grid_idx'] = grid_idx
            for p in ['node_features_list', 'model_type', 'use_edge_weights']:
                if p in summary_data: hall_row[p] = summary_data[p]
            all_summaries.append(hall_row)

        # CURVES
        if save_curve_csv and 'all_curve_data' in summary_data:
            for repeat_idx, metrics in summary_data['all_curve_data'].items():
                for metric_name, values in metrics.items():
                    all_curve_rows.append({
                        'grid_idx': grid_idx,
                        'run': repeat_idx,
                        'val_type': 'internal_hmp',
                        'metric_name': metric_name,
                        'values': values.tolist() if isinstance(values, np.ndarray) else values
                    })

        # EMBEDDINGS
        artifacts_path = results_path / f'artifacts_grid_{grid_idx}_COMBINED.pkl'
        if artifacts_path.exists():
            with open(artifacts_path, 'rb') as f:
                artifacts = pickle.load(f)
            graph_df = artifacts.get('graph_level_embeddings')
            if graph_df is not None:
                graph_df['grid_idx'] = grid_idx
                all_graph_embeddings.append(graph_df)

    summary_df = pd.DataFrame(all_summaries)
    embeddings_df = pd.concat(all_graph_embeddings, ignore_index=True) if all_graph_embeddings else pd.DataFrame()
    
    if all_curve_rows:
        curves_df = pd.DataFrame(all_curve_rows).explode('values')
        curves_df['point_idx'] = curves_df.groupby(['grid_idx', 'run', 'metric_name']).cumcount()
    else:
        curves_df = pd.DataFrame()

    return summary_df, embeddings_df, curves_df


#  Main Execution

# Locate the project root (works from the terminal and interactively) and load shared paths
_start = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
_root = next(p for p in (_start, *_start.parents) if (p / "code" / "common" / "paths.py").exists())
sys.path.insert(0, str(_root / "code" / "common"))
import paths

# Define Traditional Path
TRADITIONAL_RESULTS_DIR = paths.RES_TRAD_PICKLES

# Define list of Graph Paths to process
GRAPH_MODEL_PATHS = [
    (paths.RES_GNN_PICKLES, paths.RES_GNN_TABLES)
]

# Process Traditional Models 
print("--- Processing Traditional Models ---")
EXPORT_DIR_TRAD = paths.RES_TRAD_TABLES
EXPORT_DIR_TRAD.mkdir(parents=True, exist_ok=True)

trad_summary_df, trad_tracking_df, trad_curve_df, trad_features_df = prepare_traditional_for_r(TRADITIONAL_RESULTS_DIR)

if not trad_summary_df.empty:
    trad_summary_df.to_csv(EXPORT_DIR_TRAD / 'traditional_summary_metrics.csv', index=False)
if not trad_tracking_df.empty:
    trad_tracking_df.to_csv(EXPORT_DIR_TRAD / 'traditional_sample_tracking.csv', index=False)
if not trad_curve_df.empty:
    trad_curve_df.to_csv(EXPORT_DIR_TRAD / 'traditional_curves_tidy.csv', index=False)
if not trad_features_df.empty:
    trad_features_df.to_csv(EXPORT_DIR_TRAD / 'traditional_selected_features.csv', index=False)


# Process All Graph Models 
for current_graph_dir, export_dir in GRAPH_MODEL_PATHS:
    export_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Processing Graph Model: {current_graph_dir.name} ---")
    
    # Unpack the 3 dataframes from prepare_graph_for_r
    g_summary, g_embeddings, g_curves = prepare_graph_for_r(current_graph_dir)
    
    # Export if not empty
    if not g_summary.empty:
        g_summary.to_csv(export_dir / 'graph_summary_metrics.csv', index=False)
        print(f"Saved summary to: {export_dir}")
        
    if not g_embeddings.empty:
        g_embeddings.to_csv(export_dir / 'graph_embeddings_and_predictions.csv', index=False)
        
    if not g_curves.empty:
        g_curves.to_csv(export_dir / 'graph_curves_tidy.csv', index=False)

print("\n--- All exports complete! ---")